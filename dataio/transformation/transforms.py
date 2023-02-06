import torchvision.transforms as ts
from pprint import pprint


class Transformations:

    def __init__(self, name):
        self.name = name

        # Affine and Intensity Transformations
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.random_flip_prob = 0.0

    def get_transformation(self):
        return {
            'us': self.ultrasound_transform,
        }[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'shift_val'):        self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'):       self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'):        self.scale_val = t_opts.scale
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob

    def ultrasound_transform(self):

        train_transform = ts.Compose([ts.RandomHorizontalFlip(p=self.random_flip_prob),
                                      ts.RandomAffine(degrees=self.rotate_val,
                                                      translate=self.shift_val,
                                                      scale=self.scale_val,
                                                      fill=0,
                                                      ),
                                      ])

        test_transform = ts.Compose([])

        return {'train': train_transform, 'test': test_transform}
