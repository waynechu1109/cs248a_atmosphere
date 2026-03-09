from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple
from pathlib import Path
import slangpy as spy
from pyglm import glm
import numpy as np
from PIL import Image


@dataclass
class PhysicsBasedMaterialTextureBuf:
    albedo: spy.NDBuffer


class FilteringMethod(Enum):
    NEAREST = 0
    BILINEAR = 1
    BILINEAR_DISCRETIZED_LEVEL = 2
    TRILINEAR = 3


class BRDFType(Enum):
    LAMBERTIAN = 0
    MIRROR = 1
    GLASS = 2
    RETROREFLECTIVE = 3
    RETROREFLECTIVE_LAMBERTIAN = 4
    ATMOSPHERE = 5

@dataclass
class MaterialField[T]:
    uniform_value: T | None = None
    use_texture: bool = False
    filtering_method: FilteringMethod = FilteringMethod.NEAREST
    textures: List[np.ndarray] = field(default_factory=list)

    def __init__(
        self,
        uniform_value: T | None = None,
        use_texture: bool = False,
        filtering_method: FilteringMethod = FilteringMethod.NEAREST,
        texture_map_path: str = None,
        textures: List[np.ndarray] = None,
    ):
        self.uniform_value = uniform_value
        self.use_texture = use_texture
        self.filtering_method = filtering_method
        self.MAX_MIP_LEVELS = 8
        self.textures = textures if textures is not None else []

        # If textures are provided directly (e.g., from deserialization), use them
        if textures is not None:
            self.textures = textures
            return

        # Otherwise, load from file if provided
        if texture_map_path is not None:
            self.load_texture_from_image(texture_map_path)
        elif use_texture:
            raise ValueError(
                "Texture map path or textures array is required for texture material"
            )

    def generate_mipmaps(self, base_texture: np.ndarray) -> None:
        """Generate mipmaps from a base texture image.

        :param base_texture: The base texture as numpy array (H, W, C)
        """
        # This is level 0 mipmap. It is the original texture. Populate this list
        #  with other mipmap levels, in increasing order of levels.
        textures = [base_texture]

        # TODO: Student implementation starts here.

        current = base_texture

        while len(textures) < self.MAX_MIP_LEVELS:
            h, w, c = current.shape

            # stop if cannot downsample further
            if h < 2 or w < 2:
                break

            # ensure even dimensions
            h2 = h // 2
            w2 = w // 2

            # 2x2 box filter downsampling
            # average four pixels
            next_level = (
                current[0:2*h2:2, 0:2*w2:2, :] +  # upper left  (start: stop: step)
                current[1:2*h2:2, 0:2*w2:2, :] +  # bottom left
                current[0:2*h2:2, 1:2*w2:2, :] +  # upper right
                current[1:2*h2:2, 1:2*w2:2, :]    # bottom right
            ) * 0.25

            textures.append(next_level)
            current = next_level

        # TODO: Student implementation ends here.

        self.textures = textures

    def load_texture_from_image(self, image_path: str | Path) -> None:
        """Load a texture from an image file and generate mipmaps.

        :param image_path: Path to the image file (PNG, JPG, etc.)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        pil_img = Image.open(image_path)
        # Only keep the rgb channels
        pil_img = pil_img.convert("RGB")
        # Convert to numpy array and normalize to [0, 1]
        texture_array = np.array(pil_img).astype(np.float32) / 255.0

        # Generate mipmaps
        self.generate_mipmaps(texture_array)

    def get_this(self, offset: int) -> Tuple[Dict, int]:
        mipmaps = []
        for i in range(self.MAX_MIP_LEVELS):
            if i >= len(self.textures):
                mipmaps.append(
                    {
                        "size": [0, 0],
                        "offset": offset,
                    }
                )
            else:
                texture = self.textures[i]
                mipmaps.append(
                    {
                        "size": [texture.shape[1], texture.shape[0]],
                        "offset": offset,
                    }
                )
                offset += texture.shape[0] * texture.shape[1]
        return {
            "uniformData": self.uniform_value,
            "useTexture": self.use_texture,
            "filteringMethod": self.filtering_method.value,
            "mipmap": mipmaps,
            "totalLevels": len(self.textures),
        }, offset


@dataclass
class PhysicsBasedMaterial:
    albedo: MaterialField[glm.vec3] = field(
        default_factory=lambda: MaterialField(glm.vec3(1.0, 0.0, 1.0))
    )
    smoothness: float = 0.0
    ior: float = 1.5  # Index of Refraction (default: glass ~1.5)
    brdf_type: BRDFType = BRDFType.RETROREFLECTIVE_LAMBERTIAN
    atmosphere_enabled: bool = False
    atmosphere_rayleigh_scattering_color: glm.vec3 = field(
        default_factory=lambda: glm.vec3(0.45, 0.67, 1.0)
    )
    atmosphere_mie_scattering_color: glm.vec3 = field(
        default_factory=lambda: glm.vec3(1.0, 0.72, 0.45)
    )
    atmosphere_absorption_color: glm.vec3 = field(
        default_factory=lambda: glm.vec3(1.0, 0.55, 0.2)
    )
    atmosphere_rayleigh_density_falloff: float = 6.0
    atmosphere_mie_density_falloff: float = 1.5
    atmosphere_rayleigh_strength: float = 1.0
    atmosphere_mie_strength: float = 0.35
    atmosphere_phase_g: float = 0.35
    atmosphere_planet_radius: float = 1.0
    atmosphere_thickness: float = 0.15
    atmosphere_center: glm.vec3 = field(default_factory=lambda: glm.vec3(0.0))

    def get_this(self, offset: int) -> Tuple[Dict, int]:
        albedo_data, offset = self.albedo.get_this(offset)
        return {
            "albedo": albedo_data,
            "smoothness": self.smoothness,
            "brdfType": self.brdf_type.value,
            "ior": self.ior,
            "atmosphereEnabled": 1 if self.atmosphere_enabled else 0,
            "atmosphereRayleighScatteringColor": self.atmosphere_rayleigh_scattering_color,
            "atmosphereMieScatteringColor": self.atmosphere_mie_scattering_color,
            "atmosphereAbsorptionColor": self.atmosphere_absorption_color,
            "atmosphereRayleighDensityFalloff": self.atmosphere_rayleigh_density_falloff,
            "atmosphereMieDensityFalloff": self.atmosphere_mie_density_falloff,
            "atmosphereRayleighStrength": self.atmosphere_rayleigh_strength,
            "atmosphereMieStrength": self.atmosphere_mie_strength,
            "atmospherePhaseG": self.atmosphere_phase_g,
            "atmospherePlanetRadius": self.atmosphere_planet_radius,
            "atmosphereThickness": self.atmosphere_thickness,
            "atmosphereCenter": self.atmosphere_center,
        }, offset


def create_material_buf(
    module: spy.Module, materials: List[PhysicsBasedMaterial]
) -> Tuple[spy.NDBuffer, PhysicsBasedMaterialTextureBuf]:
    device = module.device
    material_buf = spy.NDBuffer(
        device=device,
        dtype=module.PhysicsBasedMaterial.as_struct(),
        shape=(max(len(materials), 1),),
    )
    albedo_textures = []
    cursor = material_buf.cursor()
    offset = 0
    for idx, material in enumerate(materials):
        material_data, offset = material.get_this(offset)
        for level_texture in material.albedo.textures:
            albedo_textures.append(level_texture.astype(np.float32).reshape(-1, 3))
        cursor[idx].write(material_data)
    cursor.apply()
    # Concatenate all albedo textures into a single buffer.
    np_alebdo_texture = (
        np.concatenate(albedo_textures, axis=0)
        if albedo_textures
        else np.zeros((1, 3), dtype=np.float32)
    )
    albedo_tex_buf = spy.NDBuffer(
        device=device,
        dtype=module.float3,
        shape=(np_alebdo_texture.shape[0],),
    )
    albedo_tex_buf.copy_from_numpy(np_alebdo_texture)
    physics_based_material_texture_buf = PhysicsBasedMaterialTextureBuf(
        albedo=albedo_tex_buf
    )
    return material_buf, physics_based_material_texture_buf
