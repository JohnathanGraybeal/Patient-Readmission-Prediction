import os
class Utility:
    """Contains utility methods"""
    @staticmethod
    def ClearScreen():
        """Clears the screen depending on user os"""
        command = 'clear'
        if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
            command = 'cls'
        os.system(command)