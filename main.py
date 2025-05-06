import pymrio

exio3_folder = "/tmp/mrios/automdownload/exio3"

exio_downloadlog = pymrio.download_exiobase3(
    storage_folder=exio3_folder, system="pxp", years=[2011, 2012]
)

exio1 = pymrio.parse_exiobase1(
    path=exio3_folder,
)
