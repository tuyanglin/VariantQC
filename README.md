# VariantQC
Variants Quality Control by using Gradient Boost

## Installation  
1.Download
```
git clone git@github.com:tuyanglin/VariantQC.git
```
2.install requirements
```
cd VariantQC
pip install -r requirements.txt
```
## Usage
1.You need ｓ[RTGtools](https://github.com/RealTimeGenomics/rtg-tools) to get tp.vcf.gz and fp.vcf.gz.  
2.The you can call the classification script like the following example.
```
python Classifier.py tp.vcf.gz fp.vcf.gz vcf_csv_path 
```
