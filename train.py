from typing import Tuple
import math

import torch
import torch.nn as nn
import wandb


def train(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    data_loader,
    batch_size,
    input_length,
    img_shape: tuple,
    epochs: int = 3,
) -> Tuple[nn.Module]:
    """Trains the even GAN

    Args:
        batch_size: The number of examples in a training batch
        epochs: The number of epochs to train for.

    Returns:
        generator: The trained generator model
        discriminator: The trained discriminator model
    """
    # loss is binary cross entropy loss
    loss = nn.BCELoss()
    img_w = img_shape[0]
    img_h = img_shape[1]

    for i in range(epochs):
        for sample in data_loader:
            # zero the gradients on each iteration
            generator_optimizer.zero_grad()

            # Here we create the noise input for generator and pass it through the generator to create our fake data
            noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
            generated_data = generator(noise)
            generated_data = generated_data.view(batch_size, 1, img_w, img_h) # resize to be image shape with channel 1

            # Here we get real data
            true_data = sample[0]
            # TODO: create the labels for the true data, this should be a tensor of size batch_size.
            # Remember we are doing binary classification here.
            true_labels = None

            # TODO: Train the generator
            # AKA do a forward pass and get the loss (loss function is defined above)
            # We invert the labels here and don't train the discriminator because we want the generator
            # to make things the discriminator classifies as true.
            generator_discriminator_out = None
            generator_loss = None
            
            # Notice that we do not call .step on the discriminator_optimizer
            # This is so that we do not update the parameters of the discriminator when training the generator
            generator_loss.backward()
            generator_optimizer.step()

            # TODO: Train the discriminator on the true data
            # AKA do a forward pass and get the loss on the true data
            # We don't invert the labels here, why?
            discriminator_optimizer.zero_grad()
            true_discriminator_out = None
            true_discriminator_loss = None

            # Now we do a forward pass using the fake data and get the loss with inverted labels
            # We add .detach() here so that we do not backprop into the generator when we train the discriminator
            # if you're not sure what this does, thats ok! ask us we love to answer questions
            generator_discriminator_out = discriminator(generated_data.detach())
            generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            
            discriminator_loss.backward()
            discriminator_optimizer.step()
            
            wandb.log({"Generator Loss": generator_loss, 
                    "Discriminator Loss (on real images)" : true_discriminator_loss,
                    "Discriminator Loss (on fake images)": generator_discriminator_loss,
                    "Discriminator Guess (on fake images)": torch.mean(generator_discriminator_out),
                    "Discriminator Guess (on real images)": torch.mean(true_discriminator_out)})

    return generator, discriminator


if __name__ == "__main__":
    train()
