import io
import numpy as np
import torch
from PIL import Image
from pkg_resources import resource_filename
from torchvision import transforms

from . import data_loader
from . import u2net


class BackgroundRemover:
    """
    Creates a u2-net based background remover object.
    """
    def __init__(self, small: bool = False):
        """
        Keyword arguments:
        :param small: True to use smaller model
        """

        big_model_path = resource_filename('bgrm', 'resources/u2net.pth')
        small_model_path = resource_filename('bgrm', 'resources/u2netp.pth')

        self.model_path = small_model_path if small else big_model_path
        self.net = u2net.U2NETP() if small else u2net.U2NET()

        try:
            if torch.cuda.is_available():
                self.net.load_state_dict(torch.load(self.model_path))
                self.net.to(torch.device("cuda"))
            else:
                self.net.load_state_dict(
                    torch.load(
                        self.model_path,
                        map_location="cpu",
                    )
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Make sure models are stored in resources."
            )

        self.net.eval()

    @staticmethod
    def norm_pred(d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)

        return dn

    @staticmethod
    def preprocess(image):
        label_3 = np.zeros(image.shape)
        label = np.zeros(label_3.shape[0:2])

        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        transform = transforms.Compose(
            [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
        )
        sample = transform({"imidx": np.array([0]), "image": image, "label": label})

        return sample

    def predict(self, item):
        sample = self.preprocess(item)

        with torch.no_grad():

            if torch.cuda.is_available():
                inputs_test = torch.cuda.FloatTensor(
                    sample["image"].unsqueeze(0).cuda().float()
                )
            else:
                inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

            d1, d2, d3, d4, d5, d6, d7 = self.net(inputs_test)

            pred = d1[:, 0, :, :]
            prediction = self.norm_pred(pred).squeeze()
            predict_np = prediction.cpu().detach().numpy()
            img = Image.fromarray(predict_np * 255).convert("RGB")

            del d1, d2, d3, d4, d5, d6, d7, pred, prediction, predict_np, inputs_test, sample

            return img

    @staticmethod
    def naive_cutout(img, mask):
        empty = Image.new("RGBA", img.size, "white")
        cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
        return cutout

    def remove(self, data):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        mask = self.predict(np.array(img)).convert("L")
        cutout = self.naive_cutout(img, mask)

        bio = io.BytesIO()
        cutout.save(bio, "PNG")

        return bio.getbuffer()

