import pandas as pd
import os
from torch.utils.data import Dataset

class AmphibiousIdentificationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with:
             - names.csv file
             - directory 'identified'

            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.names_frame = pd.read_csv(os.path.join(root_dir, "names.csv"))
        self.root_dir = root_dir
        self.transform = transform
        self.scan_dir()

    def scan_dir(self):
        for index,row in self.names_frame.iterrows():
            print(row['name'])
            print(row['order'])
            print()
            
            # os.listdir(os.path.join(self.root_dir, ))

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

AmphibiousIdentificationDataset('./Amphibious dataset')