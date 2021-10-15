import os
import numpy as np
import cv2
import json
from .base import PuzzleEnv


class DigitJump(PuzzleEnv):

    def __init__(self, render_style='mnist', **kwargs):
        super(DigitJump, self).__init__(render_style=render_style, **kwargs)
        self._can_go = lambda r, c: (0 <= r < self.size) and (0 <= c < self.size)

    def _move(self, r, c, d, dist):
        new_pos = (r + self.y[d] * dist, c + self.x[d] * dist)
        return new_pos if self._can_go(*new_pos) else (r, c)

    def _reset(self):
        self._create_level()
        return self.render()

    def _create_level(self):
        for _ in range(self.max_tries):
            start = (0, 0)
            end = (self.size - 1, self.size - 1)
            grid = [[self.rng.randint(1, min(6, self.size-1)) for _ in range(self.size)] for _ in range(self.size)]
            q = [(start, [])]
            n = start
            z = {}
            while q and n != end:
                n, path = q.pop()
                for d in range(4):
                    N = self._move(*n, d, grid[n[0]][n[1]])
                    if N not in z:
                        q[:0] = [(N, path + [d])]
                        z[N] = 1
            if n == end:
                break
        self.grid = np.array(grid).astype(np.uint8)
        self.pos = start
        self.end = end
        self.solution = path

        if self.render_style == 'mnist':
            data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/mnist.csv'), delimiter=',')
            self.targets = data[:, -1]
            self.images = 1. - data[:, :-1].reshape(-1, 8, 8, 1).astype(float)/16.
            self.texture = [self.images[list(np.where(self.targets == i)[0])[0]] for i in range(7)][1:]
            self.player_rgb = np.stack([0.923 * np.ones((8, 8)), 0.386 * np.ones((8, 8)), 0.209 * np.ones((8, 8))], axis=-1).astype(float) * 255.
            self.non_player_rgb = np.stack([0.56 * np.ones((8, 8)), 0.692 * np.ones((8, 8)), 0.195 * np.ones((8, 8))], axis=-1).astype(float) * 255.
        elif self.render_style == 'grid_world':
            self.player_rgb = np.pad(np.ones((6, 6)), ((1, 1), (1, 1)), mode='constant', constant_values=0).reshape((8, 8, 1)).astype(float)
            self.palette = [[132., 94., 194.], [214., 93., 177.], [255., 111., 145.], [255., 150., 113.], [255., 199., 95.], [249., 248., 113.]]
            self.palette = [np.stack([np.stack([np.array(p)]*8, axis=0)]*8, axis=0).astype(float) for p in self.palette]
        elif self.render_style == 'dice':
            data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/faces.csv'), delimiter=',')
            images = data.reshape((-1, 8, 8, 1)).astype(float)
            self.texture = [i for i in images]
            self.player_rgb = np.ones((8, 8, 3)).astype(float) * 255.
            self.palette = [[132., 94., 194.], [214., 93., 177.], [255., 111., 145.], [255., 150., 113.], [255., 199., 95.], [249., 248., 113.]]
            self.palette = [np.stack([np.stack([np.array(p)]*8, axis=0)]*8, axis=0).astype(float) for p in self.palette]
        elif self.render_style == 'beta':
            self.palette = [np.stack([np.ones((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))], axis=-1) * float(i) *(255./6.) for i in range(7)][1:]
            self.player_rgb = np.stack([np.zeros((8, 8)), np.pad(np.ones((4, 4)), ((2, 2), (2, 2)), mode='constant', constant_values=0), np.zeros((8, 8))], axis=-1).astype(float) * 255.
        else:
            raise Exception('Unknown rendering mode.')

        return self.render()

    def _step(self, a):
        if a in self.action_space and a < 4:
            self.pos = self._move(*self.pos, a, self.grid[self.pos])

        reward, done = (10., True) if (self.pos == self.end and not self.already_solved) else (0, False)
        self.already_solved = True if self.pos == self.end else self.already_solved

        return self.render(), reward, done, self.labels()

    def _get_image(self):
        if self.render_style == 'mnist':
            rgb = np.concatenate([np.concatenate([self.player_rgb * self.texture[el-1] if self.pos == (i, j) else self.non_player_rgb * self.texture[el-1]
                                                  for j, el in enumerate(row)], axis=1) for i, row in enumerate(self.grid)], axis=0)
        elif self.render_style == 'grid_world':
            rgb = np.concatenate([np.concatenate([self.palette[el-1] * self.player_rgb if self.pos == (i, j) else self.palette[el-1]
                                                  for j, el in enumerate(row)], axis=1) for i, row in enumerate(self.grid)], axis=0)
        elif self.render_style == 'dice':
            rgb = np.concatenate([np.concatenate([self.player_rgb * self.texture[el-1] if self.pos == (i, j) else self.palette[el-1] * self.texture[el-1]
                                                  for j, el in enumerate(row)], axis=1) for i, row in enumerate(self.grid)], axis=0)
            '''
            # Slightly faster (~10%) but more complex way
            a = np.array(self.grid).reshape((-1, 1, 1)) - 1
            b = tuple((p * t).reshape((1, -1, 3)) for p, t in zip(self.palette, self.texture))
            rgb = np.choose(a, b)
            rgb = rgb.reshape((self.size, self.size, 8, 8, 3))
            rgb[self.pos] = self.player_rgb * self.texture[self.grid[self.pos]-1]
            rgb = np.moveaxis(rgb, 2, 1).reshape((self.size*8, self.size*8, 3))
            '''
        elif self.render_style == 'beta':
            rgb = np.concatenate([np.concatenate([self.player_rgb + self.palette[el-1] if self.pos == (i, j) else self.palette[el-1]
                                                  for j, el in enumerate(row)], axis=1) for i, row in enumerate(self.grid)], axis=0)
        else:
            raise Exception('Unknown rendering mode.')
        rescaled = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
        render = np.clip(rescaled, 0, 255)
        return render.astype(np.uint8)
