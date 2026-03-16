import json
import inspect
import time
import uuid
import functools
import sqlite3
from pathlib import Path
import contextlib
from typing import Any, Dict, Optional, Iterator, Callable, TypeVar, Union, cast, List
import traceback
import pandas as pd
import dill
from threading import local
import os
import atexit
import threading


T = TypeVar('T', bound=Callable[..., Any])


def log_parameters(
    func: Optional[Callable] = None, 
    *,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    **metadata
) -> Union[Callable[[T], T], T]:
    """
    Decorator that logs function parameters and creates a trace ID to link all logs
    within the function call.
    """
    def _decorator(f: T) -> T:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Get function signature and bind arguments
                sig = inspect.signature(f)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Get all parameters
                params = dict(bound_args.arguments)
                
                # Extract and validate logger
                logger = params.get('logger')
                if not isinstance(logger, ExperimentLogger):
                    raise ValueError("Missing or invalid logger parameter")
                params.pop('logger')
                
                # Filter parameters based on include/exclude lists
                if include is not None:
                    params = {k: v for k, v in params.items() if k in include}
                if exclude is not None:
                    params = {k: v for k, v in params.items() if k not in exclude}
                
                # Generate trace ID for this function call
                trace_id = str(uuid.uuid4())
                
                # Log parameters with trace ID
                logger.log(
                    params,
                    trace_id=trace_id,
                    event='function_entry',
                    function_name=f.__name__,
                    **metadata
                )
                
                # Execute function with trace context
                with logger._trace_context(trace_id):
                    try:
                        return f(*args, **kwargs)
                    except Exception as function_exception:
                        traceback.print_exc()
                        logger.log(function_exception, function_name=f.__name__)
                    
            except Exception as e:
                raise RuntimeError(f"Error in parameter logging: {str(e)}") from e
            
        return cast(T, wrapper)
    
    if func is None:
        return _decorator
    return _decorator(func)


class ExperimentLogger:
    """A stateful logger that links logs within function calls using trace IDs.
    
    Uses SQLite for multi-process safe storage with proper locking and transactions.
    Optimized for multiple concurrent writers.
    """
    
    def __init__(self, 
                 log_directory: Union[str, Path] = 'experiment_logs', 
                 **metadata: Any) -> None:
        """Initialize the logger with a directory and base metadata."""
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.db_path = self.log_dir / 'experiment_logs.db'
        self.base_metadata = metadata
        
        # Thread-local storage with process ID tracking to handle forks
        self._local = local()
        self._init_process_id = os.getpid()
        
        # Connection pool: one connection per thread/process
        self._connections = {}
        
        # Initialize database schema with main connection
        self._init_database()
        
        # Register cleanup
        atexit.register(self._cleanup_connections)

    @property
    def _trace_stack(self) -> List[str]:
        """Get the trace stack for the current thread/process.
        
        Handles fork() by detecting PID changes and resetting the stack.
        """
        current_pid = os.getpid()
        
        # Detect if we've forked and reset if so
        if not hasattr(self._local, 'pid') or self._local.pid != current_pid:
            self._local.pid = current_pid
            self._local.trace_stack = []
        
        if not hasattr(self._local, 'trace_stack'):
            self._local.trace_stack = []
            
        return self._local.trace_stack

    def _init_database(self) -> None:
        """Initialize the SQLite database with the required schema.
        
        This configures WAL mode on the main connection which propagates to all future connections.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        
        # CRITICAL: Must fetch the result for WAL mode to take effect
        result = conn.execute('PRAGMA journal_mode=WAL').fetchone()
        if result[0] != 'wal':
            raise RuntimeError(f"Failed to enable WAL mode, got: {result[0]}")
        
        # Set synchronous mode for better performance with WAL
        conn.execute('PRAGMA synchronous=NORMAL')
        
        # Create schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                object_data BLOB NOT NULL,
                metadata_json TEXT NOT NULL,
                function_name TEXT,
                variable_name TEXT,
                trace_id TEXT,
                trace_stack_json TEXT,
                event TEXT
            )
        """)
        
        # Create indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON logs(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_function_name 
            ON logs(function_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_id 
            ON logs(trace_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_variable_name 
            ON logs(variable_name)
        """)
        
        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a pooled database connection for the current thread/process.
        
        Uses connection pooling to avoid recreating connections on every log call.
        """
        # Use (thread_id, process_id) as key to handle both threading and multiprocessing
        key = (threading.get_ident(), os.getpid())
        
        if key not in self._connections or self._connections[key] is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False  # We manage thread safety ourselves
            )
            
            # WAL mode is already set on the database file from _init_database
            # But we still need to verify and set other pragmas for this connection
            result = conn.execute('PRAGMA journal_mode').fetchone()
            if result[0] != 'wal':
                # Force WAL if not set (shouldn't happen, but defensive)
                conn.execute('PRAGMA journal_mode=WAL').fetchone()
            
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA busy_timeout=30000')
            
            self._connections[key] = conn
        
        return self._connections[key]

    def _cleanup_connections(self):
        """Close all pooled connections on exit."""
        for conn in self._connections.values():
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    pass
        self._connections.clear()

    @contextlib.contextmanager
    def _trace_context(self, trace_id: str):
        """Context manager for tracking the current function's trace ID."""
        self._trace_stack.append(trace_id)
        try:
            yield
        finally:
            self._trace_stack.pop()
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the actual calling function."""
        frame = inspect.currentframe()
        if frame is None:
            return {}
            
        caller = frame.f_back
        if caller is None:
            return {}
            
        # Skip logger internal frames and decorators
        while caller:
            code = caller.f_code
            if (code.co_filename == __file__ and 
                code.co_name in ['log', '_trace_context', '_decorator', 'wrapper']):
                caller = caller.f_back
            else:
                break
                
        if caller is None:
            return {}
            
        return {
            'function_name': caller.f_code.co_name,
        }

    def _get_variable_name(self, obj: Any, frame_locals: Dict[str, Any]) -> Optional[str]:
        """Get the variable name for an object in the given frame locals."""
        try:
            obj_id = id(obj)
            names = [
                name for name, value in reversed(list(frame_locals.items())) 
                if id(value) == obj_id
            ]
            return names[0] if names else None
        except Exception:
            return None

    def log(self, obj: Any, **metadata: Any) -> None:
        """Log an object with metadata, trace stack, and variable name.
        
        Uses connection pooling and deferred transactions for better concurrent write performance.
        """
        try:
            timestamp = time.time()
            obj_id = str(uuid.uuid4())  # Use UUID for guaranteed uniqueness
            
            # Get caller information
            caller_info = self._get_caller_info()
            
            # Get variable name
            if frame := inspect.currentframe():
                if caller := frame.f_back:
                    if var_name := self._get_variable_name(obj, caller.f_locals):
                        caller_info['variable_name'] = var_name
            
            # Build metadata
            combined_metadata = {
                'id': obj_id,
                'timestamp': timestamp,
                **caller_info,
                **self.base_metadata,
            }
            
            # Add trace stack if available
            trace_id = None
            trace_stack_json = None
            if self._trace_stack:
                trace_id = self._trace_stack[-1]  # Current trace ID
                combined_metadata['trace_id'] = trace_id
                combined_metadata['trace_stack'] = self._trace_stack.copy()
                trace_stack_json = json.dumps(self._trace_stack)
            
            # Add call-specific metadata
            combined_metadata.update(metadata)
            
            # Serialize the object using dill
            object_data = dill.dumps(obj)
            
            # Get pooled connection
            conn = self._get_connection()
            
            # Store in database with retry logic for concurrent writes
            # Using more retries since concurrent writes may need them
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    # Let SQLite handle transactions automatically with DEFERRED
                    # This is lighter weight than BEGIN IMMEDIATE
                    conn.execute("""
                        INSERT INTO logs 
                        (id, timestamp, object_data, metadata_json, 
                         function_name, variable_name, trace_id, trace_stack_json, event)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        obj_id,
                        timestamp,
                        object_data,
                        json.dumps(combined_metadata),
                        caller_info.get('function_name'),
                        caller_info.get('variable_name'),
                        trace_id,
                        trace_stack_json,
                        metadata.get('event')
                    ))
                    conn.commit()
                    break
                    
                except sqlite3.OperationalError as e:
                    conn.rollback()
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter for better distribution
                            import random
                            sleep_time = (0.01 * (2 ** attempt)) + (random.random() * 0.005)
                            time.sleep(sleep_time)
                        else:
                            raise RuntimeError(
                                f"Failed to write to database after {max_retries} retries. "
                                "Database may be under heavy concurrent write load."
                            ) from e
                    else:
                        raise
                    
        except Exception as e:
            raise RuntimeError(f"Error logging object: {str(e)}") from e
    
    def query(self, metadata_query: Dict[str, Any]) -> Iterator[Any]:
        """
        Query logged objects based on metadata.
        Special handling for trace_id to match anywhere in the trace stack.
        """
        for result in self.query_with_metadata(metadata_query):
            yield result['object']

    def query_with_metadata(self, metadata_query: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Query logged objects based on metadata and return both objects and their metadata.
        Special handling for trace_id to match anywhere in the trace stack.
        
        Args:
            metadata_query: Dictionary of metadata key-value pairs to match
            
        Returns:
            Iterator of dictionaries containing:
                - 'object': The stored object
                - 'metadata': The full metadata record for this object
        """
        try:
            # Build SQL query based on metadata_query
            conditions = []
            params = []
            
            # Handle special fields that have dedicated columns
            special_fields = {
                'function_name', 'variable_name', 'trace_id', 'event'
            }
            
            for key, value in metadata_query.items():
                if key in special_fields:
                    if key == 'trace_id':
                        # Special handling: match either direct trace_id or in trace_stack
                        conditions.append(
                            "(trace_id = ? OR trace_stack_json LIKE ?)"
                        )
                        params.extend([value, f'%"{value}"%'])
                    else:
                        conditions.append(f"{key} = ?")
                        params.append(value)
            
            # Build the WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, object_data, metadata_json
                FROM logs
                WHERE {where_clause}
                ORDER BY timestamp
            """
            
            conn = self._get_connection()
            cursor = conn.execute(query, params)
            
            for row in cursor:
                obj_id, timestamp, object_data, metadata_json = row
                
                # Deserialize metadata
                metadata = json.loads(metadata_json)
                
                # Additional filtering for non-special fields
                matches = True
                for key, value in metadata_query.items():
                    if key not in special_fields:
                        if metadata.get(key) != value:
                            matches = False
                            break
                
                if matches:
                    # Deserialize object
                    obj = dill.loads(object_data)
                    
                    yield {
                        'object': obj,
                        'metadata': metadata
                    }
                                
        except Exception as e:
            raise RuntimeError(f"Error querying objects: {str(e)}") from e


def load_experiment_logs(
    log_directory: Union[str, Path],
    include_trace_stack: bool = True,
    additional_explode_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load experimental logs from SQLite database into a pandas DataFrame.
    
    Args:
        log_directory: Path to the log directory containing the SQLite database
        include_trace_stack: Whether to keep and unroll the trace stack into separate columns
        additional_explode_columns: List of additional columns that contain JSON objects
            that should be exploded into separate columns
    
    Returns:
        pd.DataFrame: DataFrame containing the parsed logs
    """
    log_dir = Path(log_directory)
    db_path = log_dir / 'experiment_logs.db'
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    # Read data from SQLite
    conn = sqlite3.connect(str(db_path))
    
    # Load all metadata
    query = "SELECT metadata_json FROM logs ORDER BY timestamp"
    cursor = conn.execute(query)
    
    records = []
    for (metadata_json,) in cursor:
        try:
            record = json.loads(metadata_json)
            records.append(record)
        except json.JSONDecodeError:
            continue
    
    conn.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Handle trace information
    if 'trace_stack' in df.columns and include_trace_stack:
        # Find the maximum depth of any trace stack
        max_depth = 0
        for stack in df['trace_stack'].dropna():
            if isinstance(stack, list):
                max_depth = max(max_depth, len(stack))
        
        # Create new columns for each level of the stack
        for depth in range(max_depth):
            col_name = f'trace_stack.{depth}'
            df[col_name] = None
            
            for idx, stack in df['trace_stack'].items():
                try:
                    if pd.isna(stack) or not isinstance(stack, list):
                        continue
                except Exception:
                    continue
                if depth < len(stack):
                    df.at[idx, col_name] = stack[depth]
        
        # Drop the original trace_stack column
        df = df.drop('trace_stack', axis=1)
        
        # Add depth of call stack
        df['call_depth'] = df.filter(like='trace_stack').notna().sum(axis=1)
    
    # Explode additional JSON columns if specified
    if additional_explode_columns:
        for col in additional_explode_columns:
            if col in df.columns:
                if not isinstance(df[col].iloc[0], dict):
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                
                json_df = pd.json_normalize(df[col].dropna())
                json_df = json_df.add_prefix(f"{col}_")
                df = df.drop(col, axis=1).join(json_df)
    
    return df


def params_and_trace_ids_by_function(log_folder, metadata_df, function_to_get):
    """
    Extract parameters and trace IDs for a specific function from the logs.
    
    Args:
        log_folder: Path to the log directory
        metadata_df: DataFrame containing metadata (from load_experiment_logs)
        function_to_get: Name of the function to extract parameters for
    
    Returns:
        Tuple of (params_list, trace_ids_list)
    """
    log_dir = Path(log_folder)
    db_path = log_dir / 'experiment_logs.db'
    
    conn = sqlite3.connect(str(db_path))
    
    # Filter metadata for the specific function and variable name
    metadata_params_df = metadata_df[
        (metadata_df["variable_name"] == "params") & 
        (metadata_df["function_name"] == function_to_get)
    ]
    
    params_list = []
    trace_ids_list = metadata_params_df["trace_id"].tolist()
    
    # Retrieve objects from database
    for obj_id in metadata_params_df["id"].tolist():
        cursor = conn.execute(
            "SELECT object_data FROM logs WHERE id = ?",
            (obj_id,)
        )
        row = cursor.fetchone()
        if row:
            params_list.append(dill.loads(row[0]))
    
    conn.close()
    
    assert len(params_list) == len(set(trace_ids_list))
    return params_list, trace_ids_list