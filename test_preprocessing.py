import ast
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


def definitions():
    notebook = json.loads(Path(__file__).with_name('deepfake_detection.ipynb').read_text())
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    scope = {'os': os, 'random': random, 'Dataset': object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook definitions>', 'exec'), scope)
    return scope


class PreprocessingTests(unittest.TestCase):
    def test_two_videos_preserve_both_crops(self):
        scope = definitions()
        writes = {}
        class Capture:
            def __init__(self, path):
                self.frame = np.full((8, 8, 3), 1 if 'first' in path else 2, dtype=np.uint8)
                self.done = False
            def isOpened(self): return True
            def read(self):
                if self.done: return False, None
                self.done = True
                return True, self.frame
            def release(self): pass
        def write(path, image):
            writes[path] = image.copy()
            return True
        scope['cv2'] = SimpleNamespace(VideoCapture=Capture, cvtColor=lambda a, _: a,
                                      COLOR_BGR2RGB=0, imwrite=write)
        scope['mtcnn'] = SimpleNamespace(detect=lambda _: (np.array([[1, 1, 6, 6]]), None))
        # Execute the notebook's actual output-directory mapping as well as its processor.
        notebook = json.loads(Path(__file__).with_name('deepfake_detection.ipynb').read_text())
        source = ''.join(notebook['cells'][2]['source'])
        loop = next(node for node in ast.parse(source).body if isinstance(node, ast.For))
        with tempfile.TemporaryDirectory() as tmp:
            scope.update(INPUT_ROOT=tmp, OUTPUT_ROOT=os.path.join(tmp, 'crops'),
                         video_files=[os.path.join(tmp, 'real', f'{name}.mp4')
                                      for name in ('first', 'second')], tqdm=lambda x: x)
            exec(compile(ast.Module(body=[loop], type_ignores=[]), '<video loop>', 'exec'), scope)
        self.assertEqual(len(writes), 2)
        self.assertEqual({int(image[0, 0, 0]) for image in writes.values()}, {1, 2})

    def test_source_videos_do_not_overlap(self):
        split = definitions()['split_video_paths']
        paths, labels = [], []
        for label in (0, 1):
            for video in range(5):
                for frame in range(video + 1):
                    paths.append(f'{label}/video{video}.mp4/frame_{frame}.jpg')
                    labels.append(label)
        train, val, train_y, val_y = split(paths, labels)
        self.assertFalse(set(map(os.path.dirname, train)) & set(map(os.path.dirname, val)))
        self.assertCountEqual(train + val, paths)
        self.assertEqual(set(train_y), {0, 1})
        self.assertEqual(set(val_y), {0, 1})
        self.assertEqual(split(paths, labels), split(paths[::-1], labels[::-1]))

    def test_insufficient_videos_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'two source videos'):
            definitions()['split_video_paths'](['real/a/f.jpg', 'fake/b/f.jpg'], [0, 1])

    def test_unreadable_image_is_not_a_labeled_black_frame(self):
        scope = definitions()
        scope['cv2'] = SimpleNamespace(imread=lambda _: None)
        dataset = scope['FaceDataset'](['missing.jpg'], [1])
        with self.assertRaisesRegex(ValueError, 'missing.jpg'):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
