from pysam import VariantFile
import csv


class VCFExtract:
    """Extract data from VCF file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.vcf = VariantFile(self.filepath)
        self.header_info = self.vcf.header.info.keys()

    def fetch_data(self, outpath, label,writeHeader=True, ):
        need_info = ["BaseQRankSum", "AC", "AF", "AN", "MLEAC", "MLEAF", "FS", "InbreedingCoeff", "MQ", "MQRankSum",
                     "QD", "RAW_MQandDP", "ReadPosRankSum", "SOR", "DP"]
        info = list(set(self.header_info) & set(need_info))
        headers = ["chrom", "pos"] + info + ["label"]

        print('Start to Write csv.')
        with open(outpath, 'a+') as f:
            f_csv = csv.writer(f)
            if writeHeader:
                f_csv.writerow(headers)
            for rec in self.vcf.fetch():
                chr = rec.chrom
                pos = rec.pos
                list_info = []
                for i in info:
                    if i in rec.info:
                        if isinstance(rec.info[i], int) or isinstance(rec.info[i], float):
                            list_info.append(rec.info[i])
                        elif (isinstance(rec.info[i], tuple)):
                            list_info.append(rec.info[i][0])
                        else:
                            list_info.append(None)
                    else:
                        list_info.append(None)

                line_to_write = [chr, pos] + list_info + [label]
                f_csv.writerow(line_to_write)
                print('Write to csv Done.')

class VCF2CSV():
    def __init__(self, tp_filepath, fp_filepath):
        self.tp_vcf =VCFExtract(tp_filepath)
        self.fp_vcf = VCFExtract(fp_filepath)

    def write_to_csv(self,outpath):
        self.tp_vcf.fetch_data(outpath,1,True)
        self.fp_vcf.fetch_data(outpath, 0, False)



