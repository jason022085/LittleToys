from urllib.request import urlopen
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open
import pandas as pd


def readPDF(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content


#pdfFile = urlopen("http://pythonscraping.com/pages/warandpeace/chapter1.pdf")
pdfPath = "A systematic review on the deployment and effectiveness of data analytics in higher education to .pdf"
file = open(pdfPath, 'rb')
outputString = readPDF(file)
print(outputString)
file.close()

df_op = pd.DataFrame([outputString])
df_op.to_excel("test.xlsx")
