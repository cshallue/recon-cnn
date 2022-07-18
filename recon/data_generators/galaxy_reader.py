import asdf
from fftcorr.catalog import AbacusData, AbacusFileReader


class GalaxyReader(AbacusFileReader):
    def read(self, filename, load_velocity=False):
        with asdf.open(filename, lazy_load=True) as af:
            data = AbacusData(
                header=af.tree["header"],
                pos=af.tree["data"]["x_com"],
                weight=1.0,  # Assume all galaxies have the same mass.
                vel=af.tree["data"]["v_com"] if load_velocity else None)
        return data
