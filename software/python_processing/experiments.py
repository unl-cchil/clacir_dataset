import wesad_interfacing as wesad
import clasir_interfacing as clasir
import case_interfacing as case


window_size = 5
wesad_data, wesad_labels = wesad.windowed_feature_extraction(window_size, exclude_acc=True, dataset_name='wesad_no_acc')
clasir_data, clasir_labels = clasir.windowed_feature_extraction(window_size, exclude_acc=True, dataset_name='clasir_no_acc')
case_data, case_labels = case.windowed_feature_extraction(window_size)

# TODO: Map labels between datasets for comparison

# Perform classic benchmarking with SciKit Learn built in models on just cLASIr dataset

# Observe domain shift by using best model for each dataset on the other datasets

# Perform domain adaptation using dataset mixing, benchmark in the same way

# Perform domain adaptation using a trainable transform layer on , benchmark in the same way
