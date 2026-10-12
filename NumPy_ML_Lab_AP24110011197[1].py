# -*- coding: utf-8 -*-
"""
NumPy Hands-on ML Lab
Registration Number: AP24110011197
"""

# Q1  |  AP24110011197
import os
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([5, 10, 15, 20, 25])
print(arr)

# Q2  |  AP24110011197
zeros_array = np.zeros((5, 5))
ones_array = np.ones((2, 6))

print("5x5 Zero Array:")
print(zeros_array)

print("\n2x6 Ones Array:")
print(ones_array)

# Q3  |  AP24110011197
even_numbers = np.arange(2, 21, 2)
print(even_numbers)

evenly_spaced = np.linspace(0, 1, 5)
print(evenly_spaced)

# Q4  |  AP24110011197
a = np.arange(12)

print("2 x 6:")
print(a.reshape(2, 6))

print("\n4 x 3:")
print(a.reshape(4, 3))

print("\n6 x 2:")
print(a.reshape(6, 2))

# Q5  |  AP24110011197
matrix = np.arange(1, 10).reshape(3, 3)

print("Last row:")
print(matrix[-1])

print("\nLast column:")
print(matrix[:, -1])

print("\nMiddle element:")
print(matrix[1, 1])

print("\nBottom-right 2x2 sub-matrix:")
print(matrix[1:, 1:])

# Q6  |  AP24110011197
marks = np.array([55, 62, 48, 91, 73])

bonus_marks = marks + 1
scaled_marks = bonus_marks * 1.5

print("After adding bonus mark:")
print(bonus_marks)

print("\nScaled marks out of 150:")
print(scaled_marks)

# Q7  |  AP24110011197
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])

result = a + b

print(result)

# Q8  |  AP24110011197
marks = np.array([45, 67, 89, 32, 90, 55])

failing_marks = marks[marks < 40]
above_80_count = (marks > 80).sum()
updated_marks = np.where(marks < 40, 0, marks)

print("Failing marks:")
print(failing_marks)

print("\nNumber of students scoring above 80:")
print(above_80_count)

print("\nMarks after replacing failing marks with 0:")
print(updated_marks)

# Q9  |  AP24110011197
x = np.arange(12)

print("Split into 4 equal parts:")
print(np.split(x, 4))

print("\nAttempting to split into 5 parts:")

try:
    print(np.split(x, 5))
except ValueError as e:
    print("Error:", e)

# Q10  |  AP24110011197
from PIL import Image

try:
    # Google Colab upload
    from google.colab import files
    uploaded = files.upload()
    image_filename = list(uploaded.keys())[0]
except ImportError:
    # Local Python: place an image named "sample_image.jpg" beside this script,
    # or update this path to your image file.
    image_filename = "sample_image.jpg"
    if not os.path.exists(image_filename):
        raise FileNotFoundError(
            "Add an image named 'sample_image.jpg' beside this script, "
            "or run this section in Google Colab to upload an image."
        )

img = Image.open(image_filename).convert("RGB")
img_array = np.array(img)

print("Image shape:", img_array.shape)

# Q11  |  AP24110011197
plt.imshow(img_array)
plt.axis("off")
plt.show()

# Q12  |  AP24110011197
height, width = img_array.shape[:2]

start_y = (height - 200) // 2
start_x = (width - 200) // 2

crop = img_array[start_y:start_y + 200, start_x:start_x + 200]

plt.imshow(crop)
plt.axis("off")
plt.show()

# Q13  |  AP24110011197
vertical_flip = img_array[::-1, :]
horizontal_flip = img_array[:, ::-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_array)
axes[0].axis("off")
axes[0].set_title("Original")

axes[1].imshow(vertical_flip)
axes[1].axis("off")
axes[1].set_title("Vertical Flip")

axes[2].imshow(horizontal_flip)
axes[2].axis("off")
axes[2].set_title("Horizontal Flip")

plt.show()

# Q14  |  AP24110011197
red_channel = img_array[:, :, 0]
green_channel = img_array[:, :, 1]
blue_channel = img_array[:, :, 2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(red_channel, cmap="gray")
axes[0].axis("off")
axes[0].set_title("Red Channel")

axes[1].imshow(green_channel, cmap="gray")
axes[1].axis("off")
axes[1].set_title("Green Channel")

axes[2].imshow(blue_channel, cmap="gray")
axes[2].axis("off")
axes[2].set_title("Blue Channel")

plt.show()

# Q15  |  AP24110011197
average_brightness = np.mean(img_array)
red_brightness = np.mean(red_channel)
green_brightness = np.mean(green_channel)
blue_brightness = np.mean(blue_channel)

print("Average brightness:", average_brightness)
print("Red channel brightness:", red_brightness)
print("Green channel brightness:", green_brightness)
print("Blue channel brightness:", blue_brightness)

brightness_values = {
    "Red": red_brightness,
    "Green": green_brightness,
    "Blue": blue_brightness
}

brightest_channel = max(brightness_values, key=brightness_values.get)

print("Brightest channel:", brightest_channel)

# Q16  |  AP24110011197
gray = np.mean(img_array, axis=2)

binary_image = np.where(gray > 128, 255, 0)

plt.imshow(binary_image, cmap="gray")
plt.axis("off")
plt.show()

binary_image_100 = np.where(gray > 100, 255, 0)

plt.figure()
plt.imshow(binary_image_100, cmap="gray")
plt.axis("off")
plt.show()

binary_image_180 = np.where(gray > 180, 255, 0)

plt.figure()
plt.imshow(binary_image_180, cmap="gray")
plt.axis("off")
plt.show()

# Q17  |  AP24110011197
height, width = img_array.shape[:2]
channels = img_array.shape[2] if img_array.ndim == 3 else 1

print("Image Height:", height)
print("Image Width:", width)
print("Number of Channels:", channels)

print("\nAverage Brightness:")
print("Overall:", np.mean(img_array))
print("Red:", np.mean(img_array[:, :, 0]))
print("Green:", np.mean(img_array[:, :, 1]))
print("Blue:", np.mean(img_array[:, :, 2]))

crop_height = min(200, height)
crop_width = min(200, width)

start_y = (height - crop_height) // 2
start_x = (width - crop_width) // 2

crop = img_array[start_y:start_y + crop_height, start_x:start_x + crop_width]

vertical_flip = img_array[::-1, :]
horizontal_flip = img_array[:, ::-1]

red_channel = img_array[:, :, 0]
green_channel = img_array[:, :, 1]
blue_channel = img_array[:, :, 2]

gray = np.mean(img_array, axis=2)
thresholded = np.where(gray > 128, 255, 0)

plt.figure(figsize=(6, 6))
plt.imshow(crop)
plt.axis("off")
plt.title("Cropped Center Region")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(vertical_flip)
axes[0].axis("off")
axes[0].set_title("Vertical Flip")

axes[1].imshow(horizontal_flip)
axes[1].axis("off")
axes[1].set_title("Horizontal Flip")

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(red_channel, cmap="gray")
axes[0].axis("off")
axes[0].set_title("Red Channel")

axes[1].imshow(green_channel, cmap="gray")
axes[1].axis("off")
axes[1].set_title("Green Channel")

axes[2].imshow(blue_channel, cmap="gray")
axes[2].axis("off")
axes[2].set_title("Blue Channel")

plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(thresholded, cmap="gray")
plt.axis("off")
plt.title("Thresholded Image")
plt.show()

# Q18  |  AP24110011197
sentence = "Data Science and Machine Learning use NumPy arrays every single day"

tokens2 = sentence.lower().split()

print(tokens2)

# Q19  |  AP24110011197
tokens_array2 = np.array(tokens2)
vocabulary2 = np.unique(tokens_array2)

print("Vocabulary:", vocabulary2)
print("Vocabulary size:", len(vocabulary2))

# Q20  |  AP24110011197
def bag_of_words_vector(tokens, vocabulary):
    vector = np.array([np.sum(np.array(tokens) == word) for word in vocabulary])
    return vector

vec2 = bag_of_words_vector(tokens2, vocabulary2)

print("Vocabulary:")
print(vocabulary2)

print("\nBag-of-Words Vector:")
print(vec2)

# Q21  |  AP24110011197
sentence_a = "machine learning uses numpy arrays"
sentence_b = "deep learning also uses numpy arrays heavily"

tokens_a = sentence_a.lower().split()
tokens_b = sentence_b.lower().split()

combined_vocab = np.unique(np.array(tokens_a + tokens_b))

vec_a = bag_of_words_vector(tokens_a, combined_vocab)
vec_b = bag_of_words_vector(tokens_b, combined_vocab)

cosine_sim = np.dot(vec_a, vec_b) / (
    np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
)

print("Vector A:", vec_a)
print("Vector B:", vec_b)
print("Cosine similarity:", cosine_sim)

# Q22  |  AP24110011197
def cosine_similarity(sentence1, sentence2):
    tokens1 = sentence1.lower().split()
    tokens2 = sentence2.lower().split()

    vocabulary = np.unique(np.array(tokens1 + tokens2))

    vector1 = bag_of_words_vector(tokens1, vocabulary)
    vector2 = bag_of_words_vector(tokens2, vocabulary)

    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if denominator == 0:
        return 0

    return np.dot(vector1, vector2) / denominator

similar_sentence1 = "I like machine learning and artificial intelligence"
similar_sentence2 = "I enjoy machine learning and artificial intelligence"

different_sentence1 = "I like machine learning"
different_sentence2 = "The weather is very sunny today"

similar_score = cosine_similarity(similar_sentence1, similar_sentence2)
different_score = cosine_similarity(different_sentence1, different_sentence2)

print("Similar sentence pair score:", similar_score)
print("Different sentence pair score:", different_score)

# Q23  |  AP24110011197
paragraph = "Machine learning is useful for data science. Data science uses machine learning for analysis. Machine learning helps computers learn from data."

tokens = paragraph.lower().replace(".", "").split()

vocabulary = np.unique(np.array(tokens))

word_counts = bag_of_words_vector(tokens, vocabulary)

sorted_indices = np.argsort(word_counts)[::-1]

top_3_indices = sorted_indices[:3]

print("Top 3 most frequent words:")

for index in top_3_indices:
    print(vocabulary[index], ":", word_counts[index])

# Q24  |  AP24110011197
sentences = [
    "machine learning is useful",
    "machine learning is powerful",
    "I like eating pizza"
]

tokenized_sentences = [sentence.lower().split() for sentence in sentences]

shared_vocabulary = np.unique(
    np.array(
        [word for tokens in tokenized_sentences for word in tokens]
    )
)

vectors = [
    bag_of_words_vector(tokens, shared_vocabulary)
    for tokens in tokenized_sentences
]

similarities = {}

for i in range(3):
    for j in range(i + 1, 3):
        denominator = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])

        if denominator == 0:
            similarity = 0
        else:
            similarity = np.dot(vectors[i], vectors[j]) / denominator

        similarities[(i + 1, j + 1)] = similarity

print("Shared Vocabulary:")
print(shared_vocabulary)

print("\nBag-of-Words Vectors:")

for i, vector in enumerate(vectors, start=1):
    print("Sentence", i, ":", vector)

print("\nCosine Similarities:")

for pair, score in similarities.items():
    print("Sentence", pair[0], "and Sentence", pair[1], ":", score)

most_similar_pair = max(similarities, key=similarities.get)

print(
    "\nMost similar sentences:",
    most_similar_pair[0],
    "and",
    most_similar_pair[1]
)

print("Highest cosine similarity:", similarities[most_similar_pair])