import torch
import torch.nn as nn

class GANLoss:
    def __init__(self, lambda_mae=0.1):
        """
        Initializes the GAN loss components.
        Args:
            lambda_mae (float): Weighting factor for the MAE penalty in the generator's loss.
        """
        self.lambda_mae = lambda_mae
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.mae_loss = nn.L1Loss()

    def discriminator_loss(self, real_logits, fake_logits):
        """
        Compute the discriminator's loss.
        Args:
            real_logits (torch.Tensor): Logits predicted by the discriminator for real data.
            fake_logits (torch.Tensor): Logits predicted by the discriminator for generated data.
        Returns:
            torch.Tensor: Total loss for the discriminator.
        """
        # Real data should be classified as real (label = 1)
        real_loss = self.adversarial_loss(real_logits, torch.ones_like(real_logits))
        # Generated data should be classified as fake (label = 0)
        fake_loss = self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
        # Total discriminator loss
        total_disc_loss = (real_loss + fake_loss) / 2
        return total_disc_loss

    def generator_loss(self, fake_logits, real_data, fake_data):
        """
        Compute the generator's loss.
        Args:
            fake_logits (torch.Tensor): Logits predicted by the discriminator for generated data.
            real_data (torch.Tensor): Real data fields.
            fake_data (torch.Tensor): Generated data fields.
        Returns:
            torch.Tensor: Total loss for the generator.
        """
        # Generator aims to fool the discriminator, thus we use ones as the target labels
        adversarial_loss = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))
        # MAE penalty to enforce diversity
        mae_penalty = self.mae_loss(fake_data, real_data)
        # Total generator loss
        total_gen_loss = adversarial_loss + self.lambda_mae * mae_penalty
        return total_gen_loss

