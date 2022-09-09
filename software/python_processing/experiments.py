import wesad_interfacing as wesad
import clasir_interfacing as clasir
import case_interfacing as case


window_size = 5
wesad_data, wesad_labels = wesad.windowed_feature_extraction(window_size)
clasir_data, clasir_labels = clasir.windowed_feature_extraction(window_size)
case_data, case_labels = case.windowed_feature_extraction(window_size)

