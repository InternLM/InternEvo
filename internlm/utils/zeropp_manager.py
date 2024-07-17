import queue


class ZeroppManager:
    """
    ZeroppManager is used to manage ZeroPP parameters gathering and gradient calculation.
    """

    cache = []
    cached_wp_params = {}
    grad_queue = queue.LifoQueue()

    @classmethod
    def put(cls, total_input, grad_output, mod, communicator, func):
        # Store the weight gradient computation of linear layers.
        cls.cache.append((total_input, grad_output, mod, communicator, func))

    @classmethod
    def flush(cls):
        # Collect all stored computations during backward as a W pass.
        cls.grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def empty(cls):
        return cls.grad_queue.empty()

    @classmethod
    def pop(cls):
        item = cls.grad_queue.get()
        params = []
        for total_input, grad_output, mod, communicator, func in item:
            params.extend(func(total_input, grad_output, mod, communicator))
        return params

    @classmethod
    def cache_full_wp_parameters(cls, shard_param, full_param):
        cls.cached_wp_params[shard_param] = full_param

    @classmethod
    def retrieve_full_wp_parameters(cls, shard_param):
        return cls.cached_wp_params.get(shard_param, None)

    @classmethod
    def clear_cached_wp_parameters(cls):
        for shard_param, full_param in cls.cached_wp_params.items():
            del full_param
            cls.cached_wp_params[shard_param] = None

    @classmethod
    def check_postpond_grad_accum(cls, param):
        return param in cls.cached_wp_params
