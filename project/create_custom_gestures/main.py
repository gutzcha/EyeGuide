import customtkinter
import tkinter
from PIL import Image
import os.path as osp



customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("500x350")


def login():
    print("Hello")


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=50, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Welcome")
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="Username")
entry1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text="Password", show='x')
entry2.pack(pady=12, padx=10)

right_eye_photo = Image.open(osp.join('assets', 'right_eye.jpg'))
right_eye_photo = customtkinter.CTkImage(right_eye_photo)
# button = customtkinter.CTkButton(master=frame, text="Login", command=login, image=right_eye_photo, compound='bottom')
button = customtkinter.CTkButton(master=frame, text="", command=login, image=right_eye_photo, compound='bottom')
button.pack(pady=12, padx=10)

checkbox = customtkinter.CTkCheckBox(master=frame, text='Remember Me')
checkbox.pack()

root.mainloop()
