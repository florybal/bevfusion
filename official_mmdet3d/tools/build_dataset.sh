#!/bin/bash
set -e

DATASET=/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning

echo "======================================="
echo "Iniciando pipeline de pré-processamento do dataset BEVLOG..."
echo "======================================="

echo Diretório do dataset: $DATASET
echo "Removendo PKLs..."
rm -f $DATASET/*.pkl

echo "Removendo BINs..."
rm -rf $DATASET/bin

echo "Recriando diretório bin..."
mkdir -p $DATASET/bin

#echo "Removendo logs antigos ..."
#rm -rf /workspace/results/training/plastipak
#echo "Limpeza concluída."

echo "======================================="
echo "1. Convertendo PCD -> BIN"
echo "======================================="
python tools/build_dataset/pcd2bin.py

echo "======================================="
echo "2. Adicionando anotações XML"
echo "======================================="
python tools/build_dataset/add_annotations_from_xml.py

echo "======================================="
echo "3. Adicionando máscaras BEV"
echo "======================================="
python tools/build_dataset/process_json.py

echo "======================================="
echo "4. Gerando train/val/test"
echo "======================================="
python tools/build_dataset/split_dataset.py

echo "======================================="
echo "Pipeline finalizado!"
echo "======================================="