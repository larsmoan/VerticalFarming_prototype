import cv2
import pandas as pd
import pickle
import tkinter as tk
import datetime
import numpy as np
from PIL import Image, ImageTk

dataset_path = "/home/lars/Documents/Datasets/StrawDI_image_regression/"
labeling_file = dataset_path + "labeling_annotation.pkl"


#Information about the labeling:
# 0 -100 is the valid values for ripeness
# -1 is used as a placeholder for the images we may want to discard


class Labeler():
    def __init__(self):
        self.labeling_file = labeling_file
        self.dataset_path = dataset_path
        self.df = pd.read_pickle(labeling_file)    

        self.current_image = self.df[self.df["label"].isna()].iloc[0]["filename"]
                
    
        self.window = tk.Tk()
        self.window.title("Image labeler for ripeness")
        self.window.geometry('400x400')

        #Adding the ground truth images to the GUI, 20, 40, 60, 80, 100
        image_20 = Image.open(self.dataset_path + "ground_truth/20.png")
        image_40 = Image.open(self.dataset_path + "ground_truth/40.png")
        image_60 = Image.open(self.dataset_path + "ground_truth/60.png")
        image_80 = Image.open(self.dataset_path + "ground_truth/80.png")
        image_100 = Image.open(self.dataset_path + "ground_truth/100.png")

        #Resizing the images
        self.image_20 = self.resize_image(image_20)
        self.image_40 = self.resize_image(image_40)
        self.image_60 = self.resize_image(image_60)
        self.image_80 = self.resize_image(image_80)
        self.image_100 = self.resize_image(image_100)

        #Adding all the images to the GUI side by side
        self.image_20 = ImageTk.PhotoImage(image_20)
        self.image_40 = ImageTk.PhotoImage(image_40)
        self.image_60 = ImageTk.PhotoImage(image_60)
        self.image_80 = ImageTk.PhotoImage(image_80)
        self.image_100 = ImageTk.PhotoImage(image_100)

        #Adding the images to a common label
        self.image_label_20 = tk.Label(self.window, text="20", image=self.image_20, compound="center", font=("Helvetica", 24))
        self.image_label_40 = tk.Label(self.window, text="40", image=self.image_40, compound="center", font=("Helvetica", 24))
        self.image_label_60 = tk.Label(self.window, text="60", image=self.image_60, compound="center", font=("Helvetica", 24))
        self.image_label_80 = tk.Label(self.window, text="80", image=self.image_80, compound="center", font=("Helvetica", 24))
        self.image_label_100 = tk.Label(self.window, text="100", image=self.image_100, compound="center", font=("Helvetica", 24))

        #Packing the labels
        self.image_label_20.pack(side="left")
        self.image_label_40.pack(side="left")
        self.image_label_60.pack(side="left")
        self.image_label_80.pack(side="left")
        self.image_label_100.pack(side="left")


        #Showing the image that is going to be labeled
        image = Image.open(self.dataset_path + "images/" + self.current_image)
        width, height = image.size 
        pic_width = 200
        pic_height = 200
        scaling_factor = min(pic_width/width, pic_height/height)
        resized_image = image.resize((int(width*scaling_factor), int(height*scaling_factor)))

        self.image = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.window, image=self.image)
    
        #Make a padding between the image and the box
        self.image_label.pack(pady=20)

        self.text_box = tk.Text(self.window,font=("Helvetica", 24), height=1, width=10)
        self.text_box.pack(pady=10)
        self.text_box.bind("<Return>", self.event_handler)


        #Adding a save button
        self.save_button = tk.Button(self.window, text="Save", command=self.save)
        self.save_button.pack()

        #Print information to user about the labelling here
        print("Labeling tool for the ripeness of strawberries - use 0-100 for the actual ripeness. -1 is used for images that should be discarded.")
        
        self.window.mainloop()
    
    def save(self):
        text = self.retrieve_text()
        if text:
            self.save_label(text)
        
        #This saves the dataframe to the pickle file
        self.df.to_pickle(self.labeling_file)
        print("Saved the dataframe to: ", self.labeling_file)

        print("Non NaN values:\n", self.df.count())
        
        #This also means that the program is done, shut it down 
        self.window.destroy()
    
    def retrieve_text(self, event=None):
        text = self.text_box.get("1.0", "end-1c")
        #Clear the textbox
        self.text_box.delete("1.0", "end")
        try:
            text = int(text)
            if text not in range(-1,101):
                print("The number is not in the range -1-100, try again...")
                return False
            else:
                return text
        except ValueError:
            print("Not a valid number, try again...")
            return False

    def event_handler(self, event):
        text = self.retrieve_text(event)

        if text:
            #Save the label to the right picture
            # increment to the next image
            self.save_label(text)
            self.get_next_image()

    def get_next_image(self):
        #Gets the next image that has not been labeled
        next_filename = self.df[self.df["label"].isna()].iloc[0]["filename"]
        self.current_image = next_filename

        image = Image.open(self.dataset_path + "images/" + self.current_image)
        resized_image = self.resize_image(image)
        self.image = ImageTk.PhotoImage(resized_image)
        #Updates the image on top of the already initialized label
        self.image_label.configure(image=self.image)
             
    def save_label(self, label):
        self.df.loc[self.df["filename"] == self.current_image, "label"] = label
        return   

    def resize_image(self, image):
        #Resizes the images as close to 200x200 but keeping the aspect ratio
        width, height = image.size
        pic_width = 200
        pic_height = 200
        scaling_factor = min(pic_width/width, pic_height/height)
        resized_image = image.resize((int(width*scaling_factor), int(height*scaling_factor)))
        return resized_image

a = Labeler()
print(a.df)