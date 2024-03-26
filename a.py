import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image

# Define your dataset directory
dataset_dir = "/home/prerna/Project/images"

# Load your dataset
def load_dataset(dataset_dir):
    images = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            image_path = os.path.join(dataset_dir, filename)
            image = Image.open(image_path).convert("L")  # Convert to grayscale if needed
            image = np.array(image.resize((28, 28)))  # Resize image to 28x28 if needed
            images.append(image)
    return np.array(images)

# Load your dataset
X_train = load_dataset(dataset_dir)

# Normalize data
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], 784)  # Adjust reshape dimensions as needed

# Define dimensions
latent_dim = 100
img_dim = 784  # 28x28 pixels flattened

# Generator model
generator = Sequential([
    Dense(256, input_dim=latent_dim),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(512),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(1024),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(img_dim, activation='tanh')
])

# Discriminator model
discriminator = Sequential([
    Dense(1024, input_dim=img_dim),
    LeakyReLU(alpha=0.2),
    Dense(512),
    LeakyReLU(alpha=0.2),
    Dense(256),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')
])

# Combined model (Stacked generator and discriminator)
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# Compile models
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
generator.compile(loss='binary_crossentropy', optimizer=optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# Training
batch_size = 2  # Adjust batch size
epochs = 1000  # Adjust number of epochs

for epoch in range(epochs):
    # Generate random noise as input for the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generate fake images
    fake_images = generator.predict(noise)

    # Select a random batch of real images
    batch_indices = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
    real_images = X_train[batch_indices]

    # Concatenate real and fake images
    combined_images = np.concatenate([real_images, fake_images])

    # Labels for generated and real data
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

    # Add noise to labels
    labels += 0.05 * np.random.random(labels.shape)

    # Train discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
    g_loss = gan.train_on_batch(noise, misleading_targets)

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

# Generating and saving the images
save_dir = "./newimages"
os.makedirs(save_dir, exist_ok=True)

for i in range(6):  # Adjust number of images
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise).reshape(28, 28)
    plt.imshow(generated_image, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"generated_image_{i}.png"))
    plt.close()

print("Generated images saved successfully.")
