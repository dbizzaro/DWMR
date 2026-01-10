import random, os
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import noise
from tqdm import tqdm

#https://github.com/martius-lab/puzzlegen
from puzzlegen.puzzlegen import IceSlider


# ------------ Noise -------------
def add_gaussian_noise(x, std: float, clip_min: Optional[float] = None, clip_max: Optional[float] = None):
    """
    Add Gaussian noise with stddev `std` and clip to [clip_min, clip_max].
    Works for PIL Images (returns Image) or numpy arrays (returns ndarray).
    If clip bounds are not provided, defaults to [0,255] for PIL inputs and [0,1] for arrays.
    """
    is_pil = isinstance(x, Image.Image)
    arr = np.asarray(x, dtype=np.float32)
    if clip_min is None or clip_max is None:
        if is_pil:
            clip_min, clip_max = 0.0, 255.0
        else:
            clip_min = 0.0 if clip_min is None else clip_min
            clip_max = 1.0 if clip_max is None else clip_max
    arr = arr + np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, clip_min, clip_max)
    if is_pil:
        return Image.fromarray(arr.astype(np.uint8), mode=x.mode if hasattr(x, "mode") else None)
    return arr.astype(np.float32)


def generate_structured_noise(W, H, scale=60.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Returns a float32 array in [0, 1] with Perlin noise.
    """
    if seed is None:
        seed = np.random.randint(0, 10_000)

    # Random offsets + small random rotation fight lattice alignment
    ox, oy = np.random.uniform(0, 1e6, size=2)
    angle = np.random.uniform(0, 2*np.pi)
    ca, sa = np.cos(angle), np.sin(angle)

    # Slight anisotropy further reduces directional artifacts
    sx = scale
    sy = scale * np.random.uniform(0.9, 1.1)

    def sample(x, y):
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        amp = 1.0
        freq = 1.0
        val = 0.0
        norm = 0.0
        for _ in range(octaves):
            nx = (xr + ox) / (sx / freq)
            ny = (yr + oy) / (sy / freq)
            val += amp * noise.snoise2(nx, ny, base=seed)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return val / max(norm, 1e-8)

    field = np.zeros((H, W), dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            field[y, x] = sample(x, y)

    vmin, vmax = field.min(), field.max()
    field = (field - vmin) / (vmax - vmin + 1e-8)
    return field

def add_structured_noise(canvas_img, amplitude: float, **kw):
    W, H = canvas_img.size
    n = generate_structured_noise(W, H, **kw)   # [0,1]
    noise_arr = (n - 0.5) * 2.0 * amplitude      # [-A, A]
    base = np.asarray(canvas_img, dtype=np.float32)
    out = np.clip(base + noise_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode='L')



# ------------ 8-game and 15-game -------------

class DigitSampler:
    """Keeps a MNIST dataset and returns a fresh PIL image for a requested digit each time.
    Excludes digit 9 for n_game=8 and use EMNIST letters for n_game=15
    """
    def __init__(self, n_game: int = 8, tile_size: int = 28, split: str = "train", seed: int = 0, unique: bool = False):
        self.size = int(tile_size)
        self.root = os.path.expanduser("~/.torch_datasets")
        self.unique = unique
        self.train = split == 'train' or split == 'val'
        self.n_game = n_game
        self.split = split

        if self.unique:
            self.split = 'train'
            self.train = True
            self.seed = 0
        
        if n_game == 15:
            self.tfm = transforms.Compose([
                transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),  # rotate 90° clockwise
                transforms.Lambda(lambda img: transforms.functional.hflip(img)),        # flip horizontally
                transforms.ToTensor()
            ])
            ds = datasets.EMNIST(root=self.root, split='byclass', train=self.train, download=True, transform=self.tfm)
            self.n_classes = 26
        else:
            self.tfm = transforms.Compose([transforms.ToTensor()])
            ds = datasets.MNIST(root=self.root, train=self.train, download=True, transform=self.tfm)
            self.n_classes = 10

        if self.train:
            n_train = int(0.8 * len(ds))
            n_val = len(ds) - n_train
            train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            self.ds = train_ds if self.split == 'train' else val_ds
        else:
            self.ds = ds

        print(f"using {len(self.ds)} samples from {self.split} split")
        
        self.by_label = {d: [] for d in range(self.n_classes)}
        for i, (_, y) in enumerate(self.ds):
            yi = int(y)
            if yi in self.by_label:
                self.by_label[yi].append(i)

        self.rng = np.random.RandomState(seed)


    def sample_digit(self, d: int) -> Image.Image:
        if d not in self.by_label or len(self.by_label[d]) == 0:
            raise RuntimeError(f"No MNIST samples for digit {d}")
        if self.unique:
            idx = self.by_label[d][0]
        else:
            idx = int(self.rng.choice(self.by_label[d]))
        x, y = self.ds[idx]  # x: tensor (1,H,W) in [0,1]
        arr = (x.numpy()[0] * 255.0).astype(np.uint8)
        return Image.fromarray(arr).resize((self.size, self.size), Image.NEAREST)


def is_solvable(state: Tuple[int, ...], n_game: int = 8) -> bool:
    """
    Solvability rule for 4x4 sliding puzzle:
      Let inv = number of inversions (ignoring 0).
      Let r0 = row index of blank (0-based from top).
      Let row_from_bottom = 4 - r0 (1-based from bottom).
      Then the puzzle is solvable iff:
        - row_from_bottom is odd and inv is even, OR
        - row_from_bottom is even and inv is odd.
    """
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    if n_game == 8:
        return (inv % 2) == 0  # 3x3 solvability rule

    idx0 = state.index(0)
    r0, _ = divmod(idx0, 4)
    row_from_bottom = 4 - r0  # 1..4

    if row_from_bottom % 2 == 1:  # odd row from bottom
        return (inv % 2) == 0
    else:                         # even row from bottom
        return (inv % 2) == 1



def zero_pos(state: Tuple[int, ...], n_game: int = 8) -> Tuple[int, int]:
    idx = state.index(0)
    grid_size = 4 if n_game == 15 else 3
    return divmod(idx, grid_size)  # (row, col)



def apply_action(state: Tuple[int, ...], action: str, n_game: int = 8) -> Tuple[Tuple[int, ...], bool]:
    r, c = zero_pos(state, n_game=n_game)
    nr, nc = r, c
    if action == 'up': nr -= 1
    elif action == 'down': nr += 1
    elif action == 'left': nc -= 1
    elif action == 'right': nc += 1
    else:
        return state, False
    grid_size = 4 if n_game == 15 else 3
    if not (0 <= nr < grid_size and 0 <= nc < grid_size):
        return state, False
    new_state = list(state)
    zi = r * grid_size + c
    ni = nr * grid_size + nc
    new_state[zi], new_state[ni] = new_state[ni], new_state[zi]
    return tuple(new_state), True



def random_solvable_state(n_game: int = 8, rng: Optional[random.Random] = None) -> Tuple[int, ...]:
    if rng is None:
        rng = random.Random()
    while True:
        perm = list(range(n_game+1))
        rng.shuffle(perm)
        st = tuple(perm)
        if is_solvable(st):
            return st



def render_board(state, sampler, n_game=8, tile_px=28, pad=1, noise=0., noise_type='structured'):
    grid_size = 4 if n_game == 15 else 3
    W = grid_size*tile_px + (grid_size+1)*pad
    H = grid_size*tile_px + (grid_size+1)*pad
    canvas = Image.new('L', (W, H), color=0)
    for i, v in enumerate(state):
        r, c = divmod(i, grid_size)
        x = pad + c*(tile_px + pad)
        y = pad + r*(tile_px + pad)
        if v == 0:
            # 0 tile is a black square (visual blank), symbolic state remains 0
            tile_img = Image.new('L', (tile_px, tile_px), color=0)
        else:
            tile_img = sampler.sample_digit(int(v))
            if tile_img.size != (tile_px, tile_px):
                tile_img = tile_img.resize((tile_px, tile_px), Image.NEAREST)
        canvas.paste(tile_img, (x, y))
    if noise > 0 and noise_type == 'structured':
        canvas = add_structured_noise(canvas, noise*255.0, scale=20)
    elif noise > 0 and noise_type == 'gaussian':
        canvas = add_gaussian_noise(canvas, noise*255.0, clip_min=0.0, clip_max=255.0)
    elif noise > 0:
        raise ValueError(f"Unknown non-zero noise type: {noise_type}")
    return canvas



def sample_transitions(n, sampler, start_state=None, seed=0, noise=0., noise_type='structured', n_game=8):
    ACTIONS = ['up', 'down', 'left', 'right']
    rng = random.Random(seed)
    state = random_solvable_state(n_game, rng) if start_state is None else start_state
    pad = 1 if n_game == 8 else 0
    out = []
    for _ in range(n):
        valid = []
        for a in ACTIONS:
            ns, ok = apply_action(state, a, n_game)
            if ok:
                valid.append((a, ns))
        a, ns = rng.choice(valid)
        if sampler is not None:
            img_before = render_board(state, sampler, noise=noise, noise_type=noise_type, n_game=n_game, pad=pad)
            img_after  = render_board(ns, sampler, noise=noise, noise_type=noise_type, n_game=n_game, pad=pad)
            out.append((img_before, a, img_after, state, ns))
        else:
            out.append((state, a, ns))
        state = ns
    return out



def to_tensor(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[None, ...]  # (1, H, W)


def make_dataset(n_game=8, n = 20000, tile_size = 28, seed = 0, split = 'train', 
                 noise=0., noise_type='structured', unique_images=False):
    ACTIONS = ['up', 'down', 'left', 'right']
    ACTION_TO_ID = {a:i for i,a in enumerate(ACTIONS)}
    grid_size = 4 if n_game == 15 else 3
    sampler = DigitSampler(n_game=n_game, tile_size=tile_size, split=split, seed=seed, unique=unique_images)
    trips = sample_transitions(n, sampler, seed=seed+1, noise=noise, noise_type=noise_type, n_game=n_game)
    X_img = np.stack([to_tensor(b) for (b,_,_,_,_) in trips], axis=0)
    Y_img = np.stack([to_tensor(a) for (_,_,a,_,_) in trips], axis=0)
    X_act = np.array([ACTION_TO_ID[a] for (_,a,_,_,_) in trips], dtype=np.int64)
    S_bef = np.stack([np.array(s).reshape(grid_size,grid_size) for (_,_,_,s,_) in trips], axis=0)
    S_aft = np.stack([np.array(s).reshape(grid_size,grid_size) for (_,_,_,_,s) in trips], axis=0)
    return dict(X_img=X_img, X_act=X_act, Y_img=Y_img, S_before=S_bef, S_after=S_aft)

# ---------------------------------------------------------------------------

# ------------ Ice Slider -------------
def obs_to_tensor(obs):
    """
    Convert observation to float32 in [0,1], channel-first.
    """
    arr = np.asarray(obs)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected obs shape: {arr.shape}")
    arr = arr.transpose(2, 0, 1).astype(np.float32) # channel-first
    if arr.max() > 1.0: # normalize if looks like pixels
        arr /= 255.0
    return arr[None, ...]  # (1, C, H, W)


def make_dataset_iceslider(start_level=0, number_levels=1000, n_repeat=20, max_steps=20,
                           exclude_do_nothing=True, min_sol_len=8, noise_std = 0.0):
    """
    Generates a dataset of transitions:
      X_img, X_act, Y_img, S_before, S_after

    S_* are shaped (N, 2, 2):
      [[agent_row, agent_col],
       [goal_row,  goal_col]]

    noise_std: standard deviation of Gaussian noise added to observations (in [0,1] space).
    """

    C, H, W, G = 3, 64, 64, 8
    N = number_levels * n_repeat * max_steps
    X_img = np.empty((N, C, H, W), dtype=np.float32)
    Y_img = np.empty((N, C, H, W), dtype=np.float32)
    X_act = np.empty((N,), dtype=np.int64)
    S_before = np.empty((N, G, G), dtype=np.int64)
    S_after  = np.empty((N, G, G), dtype=np.int64)
    idx = 0
    for level_seed in tqdm(range(start_level, start_level + number_levels), desc="Levels"):
        for rep in range(n_repeat):
            env = IceSlider(seed=level_seed, min_sol_len=min_sol_len)
            obs = env.reset()
            agent_rc, goal_rc, grid = env.pos, env.end, env.grid
            grid = np.array(grid, dtype=np.int8)
            grid[goal_rc[0], goal_rc[1]] = 2  # mark goal in symbolic state
            grid[agent_rc[0], agent_rc[1]] = 3  # mark agent in symbolic state
            for t in range(max_steps):
                action = int(env.action_space.sample())
                while action == 4 and exclude_do_nothing:
                    action = int(env.action_space.sample())
                x_img = obs_to_tensor(obs)
                obs_next, _, done, info_next = env.step(action)
                y_img = obs_to_tensor(obs_next)
                if noise_std > 0.0:
                    x_img = add_gaussian_noise(x_img, noise_std, clip_min=0.0, clip_max=1.0)
                    y_img = add_gaussian_noise(y_img, noise_std, clip_min=0.0, clip_max=1.0)
                agent2_rc = env.pos
                grid2 = np.array(env.grid, dtype=np.int8)
                grid2[goal_rc[0], goal_rc[1]] = 2
                grid2[agent2_rc[0], agent2_rc[1]] = 3
                X_img[idx] = x_img
                Y_img[idx] = y_img
                X_act[idx] = action
                S_before[idx] = grid
                S_after[idx] = grid2
                idx += 1
                obs = obs_next
                agent_rc = agent2_rc
                grid = grid2
    assert idx == N, f"Collected {idx} transitions, expected {N}"
    return dict(X_img=X_img, X_act=X_act, Y_img=Y_img, S_before=S_before, S_after=S_after)

# ---------------------------------------------------------------------------

# ------------ Functionalities --------------

def show_transition(before, action, after, sbefore, safter, n_game=8, show_symbolic=True):
    if n_game == 8:
        img_size, grid_size = (88,88), (3,3)
        ACTIONS = ['up', 'down', 'left', 'right']
    elif n_game == 15:
        img_size, grid_size = (112,112), (4,4)
        ACTIONS = ['up', 'down', 'left', 'right']
    elif n_game == "ice_slider":
        img_size, grid_size = (64,64,3), (8,8)
        before = np.transpose(before, (1, 2, 0))
        after = np.transpose(after, (1, 2, 0))
        ACTIONS = ['up', 'right', 'left', 'down', 'stay']
    fig = plt.figure(figsize=(6, 2))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.imshow(before.reshape(img_size))
    if show_symbolic:
        ax1.set_title(f"State\n{np.array(sbefore).reshape(grid_size)}")
    ax1.axis('off')
    ax2.text(0.5, 0.5, f"----->\n\naction\n\"{ACTIONS[action]}\"", ha='center', va='center')
    ax2.axis('off')
    ax3.imshow(after.reshape(img_size))
    if show_symbolic:
        ax3.set_title(f"Next State\n{np.array(safter).reshape(grid_size)}")
    ax3.axis('off')
    plt.show()


def save_dataset(data: dict, path: str):
    np.savez_compressed(path, 
        X_img=data['X_img'], 
        X_act=data['X_act'], 
        Y_img=data['Y_img'], 
        S_before=data['S_before'], 
        S_after=data['S_after'])
    print(f"Dataset saved to {path}: {data['X_img'].shape[0]} samples")

def load_dataset(path: str) -> dict:
    arr = np.load(path)
    data = dict(
        X_img=arr['X_img'],
        X_act=arr['X_act'],
        Y_img=arr['Y_img'],
        S_before=arr['S_before'],
        S_after=arr['S_after'])
    print(f"Dataset loaded from {path}: {data['X_img'].shape[0]} samples")
    return data


class VisualTransitionDataset(Dataset):
    def __init__(self, dataset_dict, x_mean=None, x_std=None, standardize=True):
        super(VisualTransitionDataset, self).__init__()
        self.X_img = torch.tensor(dataset_dict['X_img'], dtype=torch.float32)
        self.X_act = torch.tensor(dataset_dict['X_act'], dtype=torch.long)
        self.Y_img = torch.tensor(dataset_dict['Y_img'], dtype=torch.float32)
        self.S_before = torch.tensor(dataset_dict['S_before'], dtype=torch.long)
        self.S_after = torch.tensor(dataset_dict['S_after'], dtype=torch.long)
        self.standardize = standardize
        self.x_mean = float(dataset_dict['X_img'].mean()) if (x_mean is None and standardize) else x_mean
        self.x_std  = float(dataset_dict['X_img'].std()) + 1e-6 if (x_std is None and standardize) else x_std
    def __len__(self):
        return self.X_img.shape[0]
    def get_mean_std(self):
        return self.x_mean, self.x_std
    def __getitem__(self, idx):
        if self.standardize:
            x = (self.X_img[idx] - self.x_mean) / self.x_std
            a = self.X_act[idx]
            y = (self.Y_img[idx] - self.x_mean) / self.x_std
            sb = self.S_before[idx]
            sa = self.S_after[idx]
            return x, a, y, sb, sa
        return self.X_img[idx], self.X_act[idx], self.Y_img[idx], self.S_before[idx], self.S_after[idx]
    


def get_dataset_splits(training_size=30000, validation_size=6000, n_game=8,
                       noise_type='none', noise_magnitude=0.5,  seeds=(42, 49, 26), unique_images=False, 
                       regenerate=False):
    '''
    Load or generate train/val/test dataset splits. n_game: 8, 15
    '''
    noise_magnitude = 0.5 if noise_type != 'none' else 0.
    unique = '_unique' if unique_images else ''
    TRAIN_DATASET_FILE = f"data/train_{n_game}_{training_size}_{noise_type}_{noise_magnitude}{unique}.npz"
    VAL_DATASET_FILE = f"data/val_{n_game}_{validation_size}_{noise_type}_{noise_magnitude}{unique}.npz"
    TEST_DATASET_FILE = f"data/test_{n_game}_{validation_size}_{noise_type}_{noise_magnitude}{unique}.npz"
    DATASET_NAME = TRAIN_DATASET_FILE + " - " + VAL_DATASET_FILE + " - " + TEST_DATASET_FILE
    if not regenerate:
        try:
            data_train = load_dataset(TRAIN_DATASET_FILE)
            data_val = load_dataset(VAL_DATASET_FILE)
            data_test = load_dataset(TEST_DATASET_FILE)
            print("Datasets found, loaded.")
        except FileNotFoundError:
            regenerate = True
    if regenerate:
        print("Generating...")
        data_train = make_dataset(n=training_size, seed=seeds[0], split='train', noise=noise_magnitude, noise_type=noise_type, n_game=n_game, unique_images=unique_images)
        save_dataset(data_train, TRAIN_DATASET_FILE)
        data_val = make_dataset(n=validation_size, seed=seeds[1], split='val', noise=noise_magnitude, noise_type=noise_type, n_game=n_game, unique_images=unique_images)
        save_dataset(data_val, VAL_DATASET_FILE)
        data_test = make_dataset(n=validation_size, seed=seeds[2], split='test', noise=noise_magnitude, noise_type=noise_type, n_game=n_game, unique_images=unique_images)
        save_dataset(data_test, TEST_DATASET_FILE)
    train_ds = VisualTransitionDataset(data_train)
    data_mean, data_std = train_ds.x_mean, train_ds.x_std
    val_ds = VisualTransitionDataset(data_val, data_mean, data_std)
    test_ds = VisualTransitionDataset(data_test, data_mean, data_std)
    return train_ds, val_ds, test_ds, DATASET_NAME


def get_dataset_splits_iceslider(training_size=40000, validation_size=8000,
                                 n_repeat=20, max_steps=20, exclude_do_nothing=True, min_sol_len=8,
                                 noise_std=0.0, regenerate=False):
    '''
    Load or generate train/val/test dataset splits for Ice Slider.
    '''
    n_game = "ice_slider"
    noise_tag = f"_noise{noise_std}" if noise_std > 0 else ""
    TRAIN_DATASET_FILE = f"data/train_{n_game}_{training_size}_{n_repeat}_{max_steps}_{exclude_do_nothing}_{min_sol_len}{noise_tag}.npz"
    VAL_DATASET_FILE = f"data/val_{n_game}_{validation_size}_{n_repeat}_{max_steps}_{exclude_do_nothing}_{min_sol_len}{noise_tag}.npz"
    TEST_DATASET_FILE = f"data/test_{n_game}_{validation_size}_{n_repeat}_{max_steps}_{exclude_do_nothing}_{min_sol_len}{noise_tag}.npz"
    DATASET_NAME = TRAIN_DATASET_FILE + " - " + VAL_DATASET_FILE + " - " + TEST_DATASET_FILE
    if not regenerate:
        try:
            data_train = load_dataset(TRAIN_DATASET_FILE)
            data_val = load_dataset(VAL_DATASET_FILE)
            data_test = load_dataset(TEST_DATASET_FILE)
            print("Datasets found, loaded.")
        except FileNotFoundError:
            regenerate = True
    if regenerate:
        print("Generating...")
        data_train = make_dataset_iceslider(start_level=0, number_levels=training_size//(n_repeat*max_steps), n_repeat=n_repeat, max_steps=max_steps, exclude_do_nothing=exclude_do_nothing, min_sol_len=min_sol_len, noise_std=noise_std)
        save_dataset(data_train, TRAIN_DATASET_FILE)
        data_val = make_dataset_iceslider(start_level=training_size//(n_repeat*max_steps), number_levels=validation_size//(n_repeat*max_steps), n_repeat=n_repeat, max_steps=max_steps, exclude_do_nothing=exclude_do_nothing, min_sol_len=min_sol_len, noise_std=noise_std)
        save_dataset(data_val, VAL_DATASET_FILE)
        data_test = make_dataset_iceslider(start_level=(training_size+validation_size)//(n_repeat*max_steps), number_levels=validation_size//(n_repeat*max_steps), n_repeat=n_repeat, max_steps=max_steps, exclude_do_nothing=exclude_do_nothing, min_sol_len=min_sol_len, noise_std=noise_std)
        save_dataset(data_test, TEST_DATASET_FILE)
    train_ds = VisualTransitionDataset(data_train, standardize=False)
    data_mean, data_std = train_ds.x_mean, train_ds.x_std
    val_ds = VisualTransitionDataset(data_val, data_mean, data_std, standardize=False)
    test_ds = VisualTransitionDataset(data_test, data_mean, data_std, standardize=False)
    return train_ds, val_ds, test_ds, DATASET_NAME
