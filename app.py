import tkinter as tk
from tkinter import filedialog

# TODO: remove later
def print_hello():
    test_output['state'] = 'normal'
    test_output.insert(tk.END, "Hello world!")
    test_output['state'] = 'disabled'

def file_prompt():
    # Open the file dialog and get the selected file path
    file_path = filedialog.askopenfilename()
    
    # Check if a file was selected
    if file_path:
        print(f"You selected: {file_path}")
        # Here, you can do something with the selected file path
    else:
        print("No file was selected")

# create and title the window
root = tk.Tk()
root.title("Instrument Recognition")

# set the size to 640 x 360
root.minsize(640, 360)
root.maxsize(640, 360)

# test button
button = tk.Button(root, text="Test Button", command=print_hello)
button.place(x=50, y=50)

# file prompt button
button= tk.Button(root, text="Input File", command=file_prompt)
button.place(x=100, y=100)

# make a text box
test_output = tk.Text(root, font=("Helvetica", "16"), width=49, height=6)
test_output.place(x=20, y=200)
test_output['state'] = 'disabled'

root.mainloop()

