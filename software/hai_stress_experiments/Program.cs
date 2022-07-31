using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MMIVR.BiosensorFramework.Extensions;
using MMIVR.BiosensorFramework.InputPipeline;
using MMIVR.BiosensorFramework.MachineLearningUtilities;


namespace hai_stress_experiments
{ 
    class Program
    {
        static string[] ExpFiles = { 
            @"C:\GitHub\hai_stress\data\raw_data\hr_data_times1.csv", 
            @"C:\GitHub\hai_stress\data\raw_data\hr_data_times2.csv" };
        static string TopDir = @"C:\GitHub\hai_stress\data\raw_data";
        static int WindowSize = 5;
        enum E4Data { ACC, BVP, EDA, HR, IBI, TAGS, TEMP }
        static void Main(string[] args)
        {
            List<ExtractedMultiFeatures> Dataset = ArceStevensDataset(TopDir);
        }
        
        public static List<Tuple<string, string>> GetSubjectConditions(string[] ExpFiles)
        {
            List<Tuple<string, string>> SubjectConditions = new List<Tuple<string, string>>();
            string line;
            List<string> cells;
            // Opening the file for reading
            foreach (string file in ExpFiles)
            {
                using (StreamReader stream = File.OpenText(file))
                {
                    stream.ReadLine();
                    while ((line = stream.ReadLine()) != null)
                    {
                        cells = line.Split(new char[] { ',' }).ToList();
                        SubjectConditions.Add(Tuple.Create(cells[0], cells[2]));
                    }
                }
            }
            return SubjectConditions;
        }

        public static List<int> GetEventIndices(Tuple<string, List<double[]>, List<string>, List<double[]>> Subject, string SubjectCondition, 
            int WindowCount)
        {
            int[] SampleRates = { 96, 32, 32, 32, 64, 4, 4 };
            List<string> DataStarts = Subject.Item3;
            List<double[]> Tags = Subject.Item4;
            List<int[]> TagIndices = new List<int[]>();
            List<int> Labels = new List<int>();

            for (int i = 0; i < DataStarts.Count; i++)
            {
                if (SampleRates[i] != 0)
                {
                    int[] indices = new int[5];
                    double StartTime = double.Parse(DataStarts[i]);
                    int PrevTimeDelta = 0;
                    for (int j = 0; j < Tags[0].Length; j++)
                    {
                        int TimeDelta = (int)(Tags[0][j] - StartTime);
                        indices[j] = (TimeDelta * SampleRates[i]) + PrevTimeDelta;
                        StartTime = Tags[0][j];
                        PrevTimeDelta = indices[j];
                    }
                    TagIndices.Add(indices);
                }
            }
            // Convert indices to seconds, assign each second a label
            int[] TestIndices = TagIndices[0].ToList().Select(x => x / 96).ToArray();

            for (int i = 0; i < WindowCount; i++)
            {
                // Before precondition
                if (i < TestIndices[0])
                {
                    Labels.Add(0);
                }
                // During precondition
                else if (TestIndices[0] <= i && i < TestIndices[1])
                {
                    Labels.Add(1);
                }
                // During condition
                else if (TestIndices[1] <= i && i < TestIndices[2])
                {
                    if (SubjectCondition == "HAI")
                        Labels.Add(2);
                    else
                        Labels.Add(3);
                }
                // During postcondition
                else if (TestIndices[2] <= i && i < TestIndices[3])
                {
                    Labels.Add(4);
                }
                // After postcondition
                else
                {
                    Labels.Add(0);
                }
            }
            return Labels;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="Directories"></param>
        public static List<ExtractedMultiFeatures> ArceStevensDataset(string Directory)
        {
            List<Tuple<string, string>> SubjectConditions = GetSubjectConditions(ExpFiles);
            List<ExtractedMultiFeatures> MultiFeatureSet = new List<ExtractedMultiFeatures>();
            List<Tuple<string, List<double[]>, List<string>, List<double[]>>> Datasets = LoadE4Dataset(Directory);
            foreach (var Dataset in Datasets)
            {
                if (SubjectConditions.Exists(c => int.Parse(c.Item1) == int.Parse(Dataset.Item1)))
                {
                    Tuple<string, string> SubjectCondition = SubjectConditions.Find(c => int.Parse(c.Item1) == int.Parse(Dataset.Item1));
                    int NumberOfSamples = Dataset.Item2[1].Length / 32;
                    List<int> EventIndices = GetEventIndices(Dataset, SubjectCondition.Item2, NumberOfSamples);
                    for (int i = 0; i < NumberOfSamples - WindowSize; i += WindowSize)
                    {
                        List<double> SubjectFeatures = new List<double>();
                        for (int j = 0; j < WindowSize; j++)
                        {
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[0].GetSubArray(j * 96, (j + 1) * 96)));
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[1].GetSubArray(j * 32, (j + 1) * 32)));
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[2].GetSubArray(j * 32, (j + 1) * 32)));
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[3].GetSubArray(j * 32, (j + 1) * 32)));
                        }
                        SubjectFeatures.AddRange(SignalProcessing.ProcessPpgSignal(Dataset.Item2[4].GetSubArray(i * 64, (i + WindowSize) * 64)));
                        SubjectFeatures.AddRange(SignalProcessing.ProcessEdaSignal(Dataset.Item2[5].GetSubArray(i * 4, (i + WindowSize) * 4)));
                        SubjectFeatures.AddRange(SignalProcessing.ProcessTmpSignal(Dataset.Item2[6].GetSubArray(i * 4, (i + WindowSize) * 4)));

                        uint label = (uint)EventIndices[i / 5];

                        MultiFeatureSet.Add(new ExtractedMultiFeatures()
                        {
                            StressFeatures = SubjectFeatures.ToArray().ToFloat(),
                            Result = label,
                        });

                        if (i + WindowSize > NumberOfSamples)
                        {
                            WindowSize = NumberOfSamples - i;
                        }
                    }
                }
            }
            return MultiFeatureSet;
        }

        /// <summary>
        /// Parses the 
        /// </summary>
        /// <param name="Directories"></param>
        /// <returns></returns>
        public static List<Tuple<string, List<double[]>, List<string>, List<double[]>>> LoadE4Dataset(string TopDir)
        {
            List<Tuple<string, List<double[]>, List<string>, List<double[]>>> Dataset = new List<Tuple<string, List<double[]>, List<string>, List<double[]>>>();
            string[] Directories = Directory.GetDirectories(TopDir);
            foreach (string Dir in Directories)
            {
                string[] directories = Directory.GetDirectories(Dir);
                foreach (string dir in directories)
                {
                    List<double[]> Data = new List<double[]>(7);
                    List<string> DataStart = new List<string>(7);
                    List<double[]> Labels = new List<double[]>();
                    List<double> LabelIndices = new List<double>();
                    string FolderName = Path.GetFileName(dir);
                    string[] files = Directory.GetFiles(dir, "*.csv").OrderBy(f => f).ToArray();
                    foreach (string file in files)
                    {
                        if (Path.GetFileNameWithoutExtension(file) == "ACC")
                        {
                            Tuple<string, double[]> AccData = ProcessE4File(file);
                            double[] AccX = new double[AccData.Item2.Length / 3],
                                AccY = new double[AccData.Item2.Length / 3],
                                AccZ = new double[AccData.Item2.Length / 3];
                            for (int i = 0; i < AccData.Item2.Length; i += 3)
                            {
                                AccX[i / 3] = AccData.Item2[i];
                                AccY[i / 3] = AccData.Item2[i + 1];
                                AccZ[i / 3] = AccData.Item2[i + 2];
                            }
                            Data.Add(AccData.Item2);
                            Data.Add(AccX);
                            Data.Add(AccY);
                            Data.Add(AccZ);
                            DataStart.Add(AccData.Item1);
                            DataStart.Add(AccData.Item1);
                            DataStart.Add(AccData.Item1);
                            DataStart.Add(AccData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "BVP")
                        {
                            Tuple<string, double[]> BvpData = ProcessE4File(file);
                            Data.Add(BvpData.Item2);
                            DataStart.Add(BvpData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "EDA")
                        {
                            Tuple<string, double[]> EdaData = ProcessE4File(file);
                            Data.Add(EdaData.Item2);
                            DataStart.Add(EdaData.Item1);
                        }
                        /*else if (Path.GetFileNameWithoutExtension(file) == "HR")
                        {
                            Tuple<string, double[]> HrData = ProcessE4File(file);
                            Data.Add(HrData.Item2);
                            DataStart.Add(HrData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "IBI")
                        {
                            Tuple<string, double[]> IbiData = ProcessE4File(file);
                            Data.Add(IbiData.Item2);
                            DataStart.Add(IbiData.Item1);
                        }*/
                        else if (Path.GetFileNameWithoutExtension(file) == "tags")
                        {
                            Tuple<string, double[]> TagData = ProcessE4File(file, true);
                            Labels.Add(TagData.Item2);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "TEMP")
                        {
                            Tuple<string, double[]> TmpData = ProcessE4File(file);
                            Data.Add(TmpData.Item2);
                            DataStart.Add(TmpData.Item1);
                        }
                    }
                    Dataset.Add(Tuple.Create(FolderName, Data, DataStart, Labels));
                }
            }
            return Dataset;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="Filepath"></param>
        /// <param name="Tags"></param>
        /// <returns></returns>
        public static Tuple<string, double[]> ProcessE4File(string Filepath, bool Tags = false)
        {
            List<string> lines = new List<string>();
            List<double> ExtractedData = new List<double>();
            string Date = "";
            // Opening the file for reading 
            using (StreamReader stream = File.OpenText(Filepath))
            {
                if (!Tags)
                {
                    Date = stream.ReadLine();
                    Date = Date.Split(new char[] { ',' }).ToList()[0];
                }
                string temp = "";
                while ((temp = stream.ReadLine()) != null)
                {
                    lines.Add(temp);
                }
            }
            foreach (string line in lines)
            {
                List<string> tempList = line.Split(new char[] { ',' }).ToList();
                foreach (string temp in tempList)
                {
                    ExtractedData.Add(double.Parse(temp));
                }
            }
            return Tuple.Create(Date, ExtractedData.ToArray());
        }
    }
}
