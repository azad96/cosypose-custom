import bokeh
import numpy as np
from PIL import Image
from itertools import cycle
import torch
import seaborn as sns
from .bokeh_utils import plot_image, to_rgba, make_image_figure, image_figure

from bokeh.models import ColumnDataSource, LabelSet


class Plotter:
    source_map = dict()

    @property
    def hex_colors(self):
        return cycle(sns.color_palette(n_colors=40).as_hex())

    @property
    def colors(self):
        return cycle(sns.color_palette(n_colors=40))

    def plot_overlay(self, rgb_input, rgb_rendered):
        rgb_input = np.asarray(rgb_input)
        rgb_rendered = np.asarray(rgb_rendered)
        assert rgb_input.dtype == np.uint8 and rgb_rendered.dtype == np.uint8
        mask = ~(rgb_rendered.sum(axis=-1) == 0)

        overlay = rgb_input
        overlay[:,:,2][mask] = overlay[:,:,2][mask]*0.80 #+ rgb_rendered[:,:,2][mask]*0.2
        overlay[:,:,1][mask] = overlay[:,:,1][mask]*0.80 + rgb_rendered[:,:,1][mask]*0.2
        overlay[:,:,0][mask] = overlay[:,:,0][mask]*0.80 #+ rgb_rendered[:,:,0][mask]*0.2
        
        f = self.plot_image(overlay, name='image')
        return f


    def plot_new_confidence_scores(self, f, detections, confidences, colors='red', line_width=0, source_id=''):
        text = ['{:.2f}'.format(conf.item()) for conf in confidences]
        boxes = None
        try:
            boxes = self.numpy(detections.bboxes)
        except:
            boxes = self.numpy(detections.initial_bboxes)
        xs = []
        ys = []
        patch_colors = []

        if text is not None:
            assert len(text) == len(boxes)
            text_x, text_y = [], []
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert len(colors) == len(boxes)
        else:
            colors = [colors for _ in range(len(boxes))]

        # Convert boxes to bokeh coordinate system
        boxes = np.array(boxes)
        boxes[:, [1, 3]] = f.plot_height - boxes[:, [1, 3]]
        for n, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            xs.append([x1, x2, x2, x1])
            ys.append([y1, y1, y2, y2])
            patch_colors.append(colors[n])
            if text is not None:
                text_x.append(x1)
                text_y.append(y1)
        source, new = self.get_source(f'{f.id}/{source_id}/bboxes')

        if new:
            f.patches(xs='xs', ys='ys', source=source,
                      line_width=line_width, color='colors', fill_alpha=0.0)

            if text is not None:
                labelset = LabelSet(x='text_x', y='text_y', text='text',
                                    text_align='left', text_baseline='bottom',
                                    text_color='white',
                                    source=source, background_fill_color='colors',
                                    text_font_size="10pt")
                f.add_layout(labelset)
        data = dict(xs=xs, ys=ys, colors=patch_colors)
        if text is not None:
            data.update(text_x=text_x, text_y=text_y, text=text)
        source.data = data
        return f


    def plot_confidence_scores(self, f, detections, pred_rendered_binary, segmentation, colors='red', line_width=0, source_id=''):
        text = []
        for det in detections:
            try:
                bbox = det.bboxes
            except:
                bbox = det.initial_bboxes
            x = round(bbox[1].item())
            y = round(bbox[0].item())
            width = round(bbox[3].item()) - x
            height = round(bbox[2].item()) - y
            obj_seg = segmentation[x:x+width, y:y+height]
            obj_rend = pred_rendered_binary[x:x+width, y:y+height]

            intersection = np.logical_and(obj_seg, obj_rend)
            union = np.logical_or(obj_seg, obj_rend)
            iou_score = np.sum(intersection) / np.sum(union)
            text.append('{:.2f}'.format(iou_score))

        boxes = None
        try:
            boxes = self.numpy(detections.bboxes)
        except:
            boxes = self.numpy(detections.initial_bboxes)
        xs = []
        ys = []
        patch_colors = []

        if text is not None:
            assert len(text) == len(boxes)
            text_x, text_y = [], []
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert len(colors) == len(boxes)
        else:
            colors = [colors for _ in range(len(boxes))]

        # Convert boxes to bokeh coordinate system
        boxes = np.array(boxes)
        boxes[:, [1, 3]] = f.plot_height - boxes[:, [1, 3]]
        for n, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            xs.append([x1, x2, x2, x1])
            ys.append([y1, y1, y2, y2])
            patch_colors.append(colors[n])
            if text is not None:
                text_x.append(x1)
                text_y.append(y1)
        source, new = self.get_source(f'{f.id}/{source_id}/bboxes')

        if new:
            f.patches(xs='xs', ys='ys', source=source,
                      line_width=line_width, color='colors', fill_alpha=0.0)

            if text is not None:
                labelset = LabelSet(x='text_x', y='text_y', text='text',
                                    text_align='left', text_baseline='bottom',
                                    text_color='white',
                                    source=source, background_fill_color='colors',
                                    text_font_size="10pt")
                f.add_layout(labelset)
        data = dict(xs=xs, ys=ys, colors=patch_colors)
        if text is not None:
            data.update(text_x=text_x, text_y=text_y, text=text)
        source.data = data
        return f

    def plot_maskrcnn_bboxes(self, f, detections, colors='red', text=None, text_auto=True, line_width=2, source_id=''):
        boxes = detections.bboxes
        if text_auto:
            text = [f'{row.label} {row.score:.2f}' for _, row in detections.infos.iterrows()]

        boxes = self.numpy(boxes)
        xs = []
        ys = []
        patch_colors = []

        if text is not None:
            assert len(text) == len(boxes)
            text_x, text_y = [], []
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert len(colors) == len(boxes)
        else:
            colors = [colors for _ in range(len(boxes))]

        # Convert boxes to bokeh coordinate system
        boxes = np.array(boxes)
        boxes[:, [1, 3]] = f.plot_height - boxes[:, [1, 3]]
        for n, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            xs.append([x1, x2, x2, x1])
            ys.append([y1, y1, y2, y2])
            patch_colors.append(colors[n])
            if text is not None:
                text_x.append(x1)
                text_y.append(y1)
        source, new = self.get_source(f'{f.id}/{source_id}/bboxes')

        if new:
            f.patches(xs='xs', ys='ys', source=source,
                      line_width=line_width, color='colors', fill_alpha=0.0)

            if text is not None:
                labelset = LabelSet(x='text_x', y='text_y', text='text',
                                    text_align='left', text_baseline='bottom',
                                    text_color='white',
                                    source=source, background_fill_color='colors',
                                    text_font_size="5pt")
                f.add_layout(labelset)
        data = dict(xs=xs, ys=ys, colors=patch_colors)
        if text is not None:
            data.update(text_x=text_x, text_y=text_y, text=text)
        source.data = data
        return f

    def _resize(self, im, size):
        return np.array(Image.fromarray(np.array(im)).resize(size))

    def _make_rgba_instance_segm(self, instance_segm, colors, alpha=0.8):
        rgba = np.zeros((*instance_segm.shape, 4), dtype=np.uint8)
        for uniq, color in zip(np.unique(instance_segm[instance_segm > 0]), colors):
            rgba[instance_segm == uniq, :3] = np.array(color) * 255
            rgba[instance_segm == uniq, -1] = alpha * 255
        return rgba

    def _make_rgba_instance_segm_colorful(self, instance_segm, colors, alpha=0.8):
        rgba = np.zeros((*instance_segm.shape[-2:], 4), dtype=np.uint8)
        for segm, color in zip(instance_segm, colors):
            rgba[segm, :3] = np.array(color) * 255
            rgba[segm, -1] = alpha * 255
        return rgba

    def plot_mask_overlay(self, im, mask, th=0.9, alpha=0.8, figure=None):
        im = self.numpy(im)
        new_fig = figure is None
        mask = self.numpy(mask)
        h, w = mask.shape
        mask_rgba = self._make_rgba_instance_segm(mask > 0.9, colors=self.colors, alpha=alpha)

        if new_fig:
            figure = make_image_figure(im_size=(w, h), axes=False)

        source, new = self.get_source(f'{figure.id}/mask')

        if new:
            figure.image_rgba('rgb', x=0, y=0, dw=w, dh=h, source=source)
            figure.image_rgba('mask', x=0, y=0, dw=w, dh=h, source=source)

        source.data = dict(rgb=[to_rgba(im)], mask=[to_rgba(mask_rgba)])
        return figure

    def masks_to_instance_segm(self, masks, thresh=0.9):
        masks = torch.as_tensor(masks).cpu().float()
        segm = torch.zeros(masks.shape[-2:], dtype=torch.uint8)
        for n, mask_n in enumerate(masks):
            m = torch.as_tensor(mask_n > thresh)
            segm[mask_n > thresh] = n + 1
        return segm

    def plot_predictions_overlay(self, figure, im, alpha=0.5):
        im = self.numpy(im)[..., :3]
        h, w, _ = im.shape

        rgba = np.zeros((*im.shape[:-1], 4), dtype=np.uint8)
        rgba[..., -1] = alpha * 255
        rgba[..., :3] = im
        rgba[im.sum(axis=-1) == 255 * 3, -1] = 120
        source, new = self.get_source(f'{figure.id}/segm')
        if new:
            figure.image_rgba('image', x=0, y=0, dw=w, dh=h, source=source)
        source.data = dict(image=[to_rgba(rgba)])
        return figure

    def plot_segm_overlay(self, im, segm, alpha=0.8, figure=None):
        if len(segm.shape) == 4:
            segm = self.masks_to_instance_segm(segm.squeeze(1))
        elif len(segm.shape) == 3:
            segm = self.masks_to_instance_segm(segm)

        im = self.numpy(im)[..., :3]
        h, w, _ = im.shape
        new_fig = figure is None
        segm = self.numpy(segm)
        if segm.dtype != np.uint8:
            segm = segm.argmax(0)

        segm_rgba = self._make_rgba_instance_segm(segm, colors=self.colors, alpha=alpha)

        if new_fig:
            figure = make_image_figure(im_size=(w, h), axes=False)

        source, new = self.get_source(f'{figure.id}/segm')

        if new:
            figure.image_rgba('rgb', x=0, y=0, dw=w, dh=h, source=source)
            figure.image_rgba('segm', x=0, y=0, dw=w, dh=h, source=source)

        source.data = dict(rgb=[to_rgba(im)], segm=[to_rgba(segm_rgba)])
        return figure

    def make_colormaps(self, instance_count):
        colors_hex = sns.color_palette(n_colors=instance_count).as_hex()
        colormap_hex = [color for color in colors_hex]
        colormap_rgb = [[int(h[1:][i:i+2], 16) / 255. for i in (0, 2, 4)] for h in colormap_hex]
        return colormap_rgb, colormap_hex

    def plot_segm_overlay_colorful(self, im, segm, alpha=0.8, figure=None):
        im = self.numpy(im)[..., :3]
        h, w, _ = im.shape
        new_fig = figure is None
        segm = self.numpy(segm)
        
        instance_count = len(segm)
        colormap_rgb, _ = self.make_colormaps(instance_count)
        segm_rgba = self._make_rgba_instance_segm_colorful(segm, colors=colormap_rgb, alpha=alpha)

        if new_fig:
            figure = make_image_figure(im_size=(w, h), axes=False)

        source, new = self.get_source(f'{figure.id}/segm')

        if new:
            figure.image_rgba('rgb', x=0, y=0, dw=w, dh=h, source=source)
            figure.image_rgba('segm', x=0, y=0, dw=w, dh=h, source=source)

        source.data = dict(rgb=[to_rgba(im)], segm=[to_rgba(segm_rgba)])
        return figure

    def plot_image(self, im, figure=None, name='image'):
        im = self.numpy(im)
        if im.shape[0] == 3:
            im = im.transpose((1, 2, 0))
        im = im[..., :4]
        if im.dtype == np.float32:
            im = (im * 255).astype(np.uint8)
        h, w, _ = im.shape
        new_fig = figure is None

        if new_fig:
            figure = make_image_figure(im_size=(w, h), axes=False)

        source, new = self.get_source(f'{figure.id}/{name}')

        if new:
            figure.image_rgba('image', x=0, y=0, dw=w, dh=h, source=source)

        source.data = dict(image=[to_rgba(im)])
        return figure

    def numpy(self, x):
        return torch.as_tensor(x).cpu().numpy()

    def set_source(self, source, name):
        self.source_map[name] = source

    def get_source(self, name):
        if name in self.source_map:
            source = self.source_map[name]
            new = False
        else:
            source = ColumnDataSource()
            self.source_map[name] = source
            new = True
        return source, new


