
import torch
from torch.utils.data import Dataset








#--------- AUGMENTATION HELPERS -----------------------------------------------------------
#---------------------------
class Augmentation(Dataset):
    """
    Class that takes in original dataset and creates a transformed
    copy whose transformation is invariant or equivariant.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if self.transform is None:
            return sample
        return self.transform(sample)
#------------------------------------




#--------------------
def b_parity(sample):
    """
    Magnetic field parity. 
    Flips the sign of the magnetic field.
    """
    X_in = sample["input_fields"].clone()
    X_out = sample["output_fields"].clone()

    # Flipping the sign of the magnetic field
    X_in[..., 1:4] *= -1
    X_out[..., 1:4] *= -1

    new_sample = dict(sample)
    new_sample["input_fields"] = X_in
    new_sample["output_fields"] = X_out
    return new_sample
#--------------------




#---------------------
def rotation_symmetry(
            sample, 
            axis_rot: str, 
            num_90_rot: int
    ):
    """
    Applies a cube-compatible rotation to one sample.
    axis_rot: 'x', 'y', or 'z'
    num_90_rot: 1, 2, or 3
    """
    #----------------------------------
    # Raising errors for faulty inputs:
    if axis_rot == "x":
        dims = (2, 3)
    elif axis_rot == "y":
        dims = (1, 3)
    elif axis_rot == "z":
        dims = (1, 2)
    else:
        raise ValueError("axis_rot must be 'x', 'y', or 'z'")

    if num_90_rot not in {1, 2, 3}:
        raise ValueError("num_90_rot must be 1, 2, or 3")
    #----------------------------------------------------

    #--------------------
    def rotate_sample(X):
        """
        Rotation for individual sample
        """
        # X shape: [T, X, Y, Z, 7]
        rho = X[..., 0:1]
        B   = X[..., 1:4]
        v   = X[..., 4:7]
        
        # Rotating vectors:
        rho_rot = torch.rot90(rho, k=num_90_rot, dims=dims)
        B_spat  = torch.rot90(B,   k=num_90_rot, dims=dims)
        v_spat  = torch.rot90(v,   k=num_90_rot, dims=dims)

        Bx = B_spat[..., 0:1]
        By = B_spat[..., 1:2]
        Bz = B_spat[..., 2:3]

        vx = v_spat[..., 0:1]
        vy = v_spat[..., 1:2]
        vz = v_spat[..., 2:3]

        # Rotating grid:
        if axis_rot == "z":
            if num_90_rot == 1:
                B_rot = torch.cat([-By,  Bx, Bz], dim=-1)
                v_rot = torch.cat([-vy,  vx, vz], dim=-1)
            elif num_90_rot == 2:
                B_rot = torch.cat([-Bx, -By, Bz], dim=-1)
                v_rot = torch.cat([-vx, -vy, vz], dim=-1)
            else:  # num_90_rot == 3
                B_rot = torch.cat([ By, -Bx, Bz], dim=-1)
                v_rot = torch.cat([ vy, -vx, vz], dim=-1)

        elif axis_rot == "x":
            if num_90_rot == 1:
                B_rot = torch.cat([Bx, -Bz,  By], dim=-1)
                v_rot = torch.cat([vx, -vz,  vy], dim=-1)
            elif num_90_rot == 2:
                B_rot = torch.cat([Bx, -By, -Bz], dim=-1)
                v_rot = torch.cat([vx, -vy, -vz], dim=-1)
            else: # num_90_rot == 3
                B_rot = torch.cat([Bx,  Bz, -By], dim=-1)
                v_rot = torch.cat([vx,  vz, -vy], dim=-1)

        else: # axis_rot == "y"
            if num_90_rot == 1:
                B_rot = torch.cat([ Bz, By, -Bx], dim=-1)
                v_rot = torch.cat([ vz, vy, -vx], dim=-1)
            elif num_90_rot == 2:
                B_rot = torch.cat([-Bx, By, -Bz], dim=-1)
                v_rot = torch.cat([-vx, vy, -vz], dim=-1)
            else:  # num_90_rot == 3
                B_rot = torch.cat([-Bz, By,  Bx], dim=-1)
                v_rot = torch.cat([-vz, vy,  vx], dim=-1)

        return torch.cat([rho_rot, B_rot, v_rot], dim=-1)
    #----------------------------------------------------

    X_in = rotate_sample(sample["input_fields"].clone())
    X_out = rotate_sample(sample["output_fields"].clone())
    new_sample = dict(sample)
    new_sample["input_fields"] = X_in
    new_sample["output_fields"] = X_out


    return new_sample
#--------------------
#------------------------------------------------------------------------------------------







