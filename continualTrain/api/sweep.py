class SweepBase:
    def __init__(self, *args, **kwargs):
        # Bypass the overridden __setattr__ for initial setup
        super().__setattr__("main_plugin", kwargs["main_plugin"])
        kwargs.pop("main_plugin")
        super().__init__(*args, **kwargs)

    def set_plugin_attribute(self, name, value):
        """Sets an attribute both on the strategy and the plugin."""
        setattr(self, name, value)  # Set attribute on strategy
        setattr(self.main_plugin, name, value)  # Set attribute on plugin
