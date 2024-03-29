import tkinter as tk

root = tk.Tk()
root.title("<Title>")

# set the size to 640 x 360
root.minsize(640, 360)
root.maxsize(640, 360)

button = tk.Button(root, text="Test Button")
button.place(x=50, y=50)

root.mainloop()

