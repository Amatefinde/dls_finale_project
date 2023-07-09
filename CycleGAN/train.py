import numpy as np
import itertools

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils import *
from cyclegan import *
import argparse


PATH_TO_DATASET = r"E:\all\studying\4 semester\dls_finale_project\Mone"


parser = argparse.ArgumentParser()
parser.add_argument("--CONTINUE_LEARNING", type=bool, default=False, help="use already existing model with weights")
args = parser.parse_args()
CONTINUE_LEARNING = args.CONTINUE_LEARNING
print(CONTINUE_LEARNING)


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    decay_start_epoch=0,
    lr=0.00002,
    b1=0.5,
    b2=0.999,
    dataset_train_name="train",
    dataset_test_name="test",
    batch_size=6,
    n_cpu=12,
    img_size=128,
    channels=3,
    n_critic=5,
    num_residual_blocks=19,
    lambda_cyc=10.0,
    lambda_id=4.0,
)

cuda_is_available = True if torch.cuda.is_available() else False
TensorDevice = torch.cuda.FloatTensor if cuda_is_available else torch.Tensor


def to_img(x):
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


transforms_ = [
    transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    ImageDataset(PATH_TO_DATASET, mode=hp.dataset_train_name, transforms_=transforms_),
    batch_size=hp.batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    ImageDataset(PATH_TO_DATASET, mode=hp.dataset_test_name, transforms_=transforms_),
    batch_size=16,
    shuffle=True,
)


def save_img_samples(batches_done):
    imgs = next(iter(val_dataloader))
    Gen_AB.eval()
    Gen_BA.eval()

    real_A = Variable(imgs["A"].type(TensorDevice))
    fake_B = Gen_AB(real_A)
    real_B = Variable(imgs["B"].type(TensorDevice))
    fake_A = Gen_BA(real_B)

    real_A = make_grid(real_A, nrow=16, normalize=True)
    real_B = make_grid(real_B, nrow=16, normalize=True)
    fake_A = make_grid(fake_A, nrow=16, normalize=True)
    fake_B = make_grid(fake_B, nrow=16, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    path = PATH_TO_DATASET + "/after_%s_batches.png" % (batches_done)

    save_image(image_grid, path, normalize=False)
    return path


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[
                        i
                    ] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

input_shape = (hp.channels, hp.img_size, hp.img_size)
if CONTINUE_LEARNING:
    Gen_AB = torch.load("weights/Gen_AB_weight.pth")
    Gen_BA = torch.load("weights/Gen_BA_weight.pth")
    Disc_A = torch.load("weights/Disc_A_weight.pth")
    Disc_B = torch.load("weights/Disc_B_weight.pth")
else:
    Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)
    Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks)
    Disc_A = Discriminator(input_shape)
    Disc_B = Discriminator(input_shape)

if cuda_is_available:
    Gen_AB = Gen_AB.cuda()
    Gen_BA = Gen_BA.cuda()
    Disc_A = Disc_A.cuda()
    Disc_B = Disc_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

fake_A_buffer = ReplayBuffer()

fake_B_buffer = ReplayBuffer()

optimizer_G = torch.optim.AdamW(
    itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()),
    lr=hp.lr,
    betas=(hp.b1, hp.b2),
)
optimizer_Disc_A = torch.optim.Adam(Disc_A.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))

optimizer_Disc_B = torch.optim.Adam(Disc_B.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LearningRateScheduler(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step
)

lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_A,
    lr_lambda=LearningRateScheduler(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)

lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_B,
    lr_lambda=LearningRateScheduler(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)



def train(
    Gen_B2A,
    Gen_A2B,
    Disc_A,
    Disc_B,
    train_dataloader,
    n_epochs,
    criterion_identity,
    criterion_cycle,
    lambda_cyc,
    criterion_GAN,
    optimizer_G,
    fake_A_buffer,
    fake_B_buffer,
    optimizer_Disc_A,
    optimizer_Disc_B,
    TensorDevice,
    lambda_id,
):
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):
            real_A = Variable(batch["A"].type(TensorDevice))
            real_B = Variable(batch["B"].type(TensorDevice))

            valid = Variable(
                TensorDevice(np.ones((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            fake = Variable(
                TensorDevice(np.zeros((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            Gen_A2B.train()
            Gen_B2A.train()

            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(Gen_B2A(real_A), real_A)
            loss_id_B = criterion_identity(Gen_A2B(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            fake_B = Gen_A2B(real_A)
            loss_GAN_AB = criterion_GAN(Disc_B(fake_B), valid)
            fake_A = Gen_B2A(real_B)
            loss_GAN_BA = criterion_GAN(Disc_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            reconstructed_A = Gen_B2A(fake_B)
            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)
            reconstructed_B = Gen_A2B(fake_A)
            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()

            optimizer_G.step()

            ################################
            optimizer_Disc_A.zero_grad()
            loss_real = criterion_GAN(Disc_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(Disc_A(fake_A_.detach()), fake)
            loss_Disc_A = (loss_real + loss_fake) / 2
            loss_Disc_A.backward()
            optimizer_Disc_A.step()
            ###############################
            optimizer_Disc_B.zero_grad()
            loss_real = criterion_GAN(Disc_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(Disc_B(fake_B_.detach()), fake)
            loss_Disc_B = (loss_real + loss_fake) / 2
            loss_Disc_B.backward()
            optimizer_Disc_B.step()
            ###############################

            loss_D = (loss_Disc_A + loss_Disc_B) / 2
            batches_done = epoch * len(train_dataloader) + i

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [DiscriminatorLoss: %f] [GeneratorLoss: %f, adv: %f, cycle: %f, identity: %f]"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                )
            )

        save_img_samples(batches_done)
        torch.save(Gen_B2A, 'weights/Gen_BA_weight.pth')
        torch.save(Gen_A2B, 'weights/Gen_AB_weight.pth')
        torch.save(Disc_A, 'weights/Disc_A_weight.pth')
        torch.save(Disc_B, 'weights/Disc_B_weight.pth')

        example_input = torch.empty([1, 3, 256, 455]).cuda() if cuda_is_available else torch.empty([1, 3, 256, 455])
        traced_model = torch.jit.trace(Gen_B2A, example_input)
        traced_model.save("generators_for_inference/Gen_B2A.pt")

        traced_model = torch.jit.trace(Gen_A2B, example_input)
        traced_model.save("generators_for_inference/Gen_A2B.pt")


if __name__ == '__main__':
    train(
        Gen_B2A=Gen_BA,
        Gen_A2B=Gen_AB,
        Disc_A=Disc_A,
        Disc_B=Disc_B,
        train_dataloader=train_dataloader,
        n_epochs=hp.n_epochs,
        criterion_identity=criterion_identity,
        criterion_cycle=criterion_cycle,
        lambda_cyc=hp.lambda_cyc,
        criterion_GAN=criterion_GAN,
        optimizer_G=optimizer_G,
        fake_A_buffer=fake_A_buffer,
        fake_B_buffer=fake_B_buffer,
        optimizer_Disc_A=optimizer_Disc_A,
        optimizer_Disc_B=optimizer_Disc_B,
        TensorDevice=TensorDevice,
        lambda_id=hp.lambda_id,
    )
