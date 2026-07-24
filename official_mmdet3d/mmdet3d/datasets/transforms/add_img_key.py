from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class AddImgKey(BaseTransform):
    """Adiciona a chave 'img' a partir de 'images' (cópia)."""
    def transform(self, results):
        if 'images' in results:
            results['img'] = results['images']
        return results