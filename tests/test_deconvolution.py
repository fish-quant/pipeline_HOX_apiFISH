from deconvolution.deconvolve import deconvolution
import os


def test_is_deconwolf_installed():
    dc = deconvolution()
    assert dc.is_deconwolf_installed()


def test_create_psf_dw():
    dc = deconvolution()
    dc.create_psf_dw(100, 100, 1.4, 1, 460, "test")
    assert os.path.exists("test.tif")
