import tkinter as tk

# TODO: remove later
def print_hello():
    print("Hello world!")

root = tk.Tk()
root.title("<Title>") #TODO: replace with app name

# set the size to 640 x 360
root.minsize(640, 360)
root.maxsize(640, 360)

button = tk.Button(root, text="Test Button", command=print_hello)
button.place(x=50, y=50)

test_output = tk.Text(root, width=20, height=5)
test_output.place(x = 100, y = 100)

root.mainloop()

