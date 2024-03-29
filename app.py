import tkinter as tk

# TODO: remove later
def print_hello():
    print("Hello world!")

# create and title the window
root = tk.Tk()
root.title("Instrument Recognition")

# set the size to 640 x 360
root.minsize(640, 360)
root.maxsize(640, 360)

# test button
button = tk.Button(root, text="Test Button", command=print_hello)
button.place(x=50, y=50)

# make a text box
test_output = tk.Text(root, font=("Helvetica", "16"), width=49, height=6)
test_output.place(x=20, y=200)

root.mainloop()

