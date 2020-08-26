![](app_demo2.gif)

# Selfie Popularity Predictor
## Motivation
The goal of my final project at Metis was to build the framework for an application, or potential "add on," to Instragram which would allow users to upload
selfies to the app and understand whether or not the photo will be popular or not. Instagram, as well as all social media platforms, are constantly striving
for our attention, and this application offers a fun game users could play as they try to up the probability their photo will be popular.

# Data
The [data](https://www.crcv.ucf.edu/data/Selfie/) was collected from a study conducted at the University of Central Florida. 
It contains over 46,000 selfies from Instagram, each containing several interesting labels. I used the provided 'log2 normalized popularity score' to determine if a photo was popular or not. Unfortunately I don't know how this score was determined, but I imagine it is related to the number of likes each photo received, normalized over the number of followers they have. Photos which scored in the top 50% were considered to be popular. 

## Image Prep
Each image was 306 by 306 pixels. To improve model performance, I resized them to be 128 by 128.

## Expression Detection
In addition to determining popularity, I wanted to return some information about each photo. 
I used [Facial Expression Recognition](https://pypi.org/project/fer/) (FER) to extract some basic emotions seen in the photos.

## Code and Initial Modeling
The code for my data cleaning and expression detection can be found in `pre_NNet_analysis.ipynb`. 
I also fit some basic classification models, none of which yielded good results, so long as I did not use the image data (pixels).

# Modeling
I used a Convolutional Neural Net, fitting the data to a sequential model. I started on a subset of my photos and once I
was comfortable with the results, I fit the model to my entire dataset. My best validation accuracy was 72%, on a nearly perfectly
balanced dataset, and I achieved an accuracy of 71% on the test set.

## Project Code
The code for my model can be found in `cnn_model.ipynb` as well as the results of my test set.

# Flask App
I built a basic Flask App (shown above) to display the results of the model. The app takes in an image of any size and returns the probability
of the photo being popular on Instragram as well as the expressions being conveyed in the photo.

# Conclusion
Given the model's accuracy, the short time I had to work on this, and the subjectivity of the target (popularity) I believe
that this result is surprisingly good. With more computing power, data (perhaps a million photos) and more time, then
it's feasible to think that Instagram popularity can be predicted with a greater accuracy than achieved here.

