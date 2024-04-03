import tkinter as tk
from tkinter import filedialog

# TODO: remove later
def print_hello():
    test_output['state'] = 'normal'
    test_output.insert(tk.END, "Hello world!")
    test_output['state'] = 'disabled'

def file_prompt():
    file_path = filedialog.askopenfilename()
    input_output_box['state'] = 'normal'
    # Check if a file was selected
    if file_path:
        split_path = file_path.split("/", -1)
        file = split_path[len(split_path) - 1]
        input_output_box.insert(tk.END, f"File selected: {file}")
    else:
        input_output_box.insert(tk.END, "No file was selected")
    input_output_box['state'] = 'disabled'

# create and title the window
root = tk.Tk()
root.title("Instrument Recognition")

# set the size to 640 x 360
root.minsize(640, 360)
root.maxsize(640, 360)

# test button
button = tk.Button(root, text="Test Button", command=print_hello)
button.place(x=20, y=180)

# file prompt button
button = tk.Button(root, text="Input File", command=file_prompt)
button.place(x=20, y=20)

# file input text box
input_output_box = tk.Text(root, font=("Helvetica", "12"), width=65, height=4)
input_output_box.place(x=20, y=60)
input_output_box['state'] = 'disabled'

# make a text box
test_output = tk.Text(root, font=("Helvetica", "16"), width=49, height=4)
test_output.place(x=20, y=220)
test_output['state'] = 'disabled'

root.mainloop()

