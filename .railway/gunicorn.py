def post_worker_init(worker):
    # Called just after a worker has been initialized
    worker.log.info("Worker spawned (pid: %s)", worker.pid)

workers = 4
worker_class = "gthread"  # Use the 'gthread' worker class for better performance
threads = 4  # Number of threads per worker

# Optional: If you're using any frameworks, you may want to tweak the worker timeout
timeout = 120

# Optional: Logging configuration
accesslog = "-"  # Prints the access log to stdout
errorlog = "-"  # Prints the error log to stdout
loglevel = "info"  # Adjust the log level as needed
