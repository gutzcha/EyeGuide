import customtkinter
from demo.rules import Blinking, SingleActionTracker
import os.path as ops


path_to_assets = 'assets'

class ActionTrackerIcon(customtkinter.CTkButton):
    def __init__(self, master, action_tracker_object: SingleActionTracker, image_path):
        super().__init__(master=master, )



if __name__ == "__main__":
    app = App()
    app.mainloop()