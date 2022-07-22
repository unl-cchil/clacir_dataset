using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MMIVR.BiosensorFramework.MachineLearningUtilities;

namespace hai_stress_experiments
{ 
    class Program
    {
        enum E4Data { ACC, BVP, EDA, HR, IBI, TAGS, TEMP }
        static void Main(string[] args)
        {
            string TopDir = @"C:\GitHub\hai_stress\data\raw_data";
            var Dataset = LoadE4Dataset(TopDir);
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="Directories"></param>
        public static void ArceStevensDatasetPipeline(string[] Directories)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="Directories"></param>
        /// <returns></returns>
        public static List<Tuple<string, List<double[]>, List<string>>> LoadE4Dataset(string TopDir)
        {
            List<Tuple<string, List<double[]>, List<string>>> Dataset = new List<Tuple<string, List<double[]>, List<string>>>();
            string[] Directories = Directory.GetDirectories(TopDir);
            foreach (string Dir in Directories)
            {
                string[] directories = Directory.GetDirectories(Dir);
                foreach (string dir in directories)
                {
                    List<double[]> Data = new List<double[]>(7);
                    List<string> Labels = new List<string>(7);
                    string FolderName = Path.GetFileName(dir);
                    string[] files = Directory.GetFiles(dir, "*.csv").OrderBy(f => f).ToArray();
                    foreach (string file in files)
                    {
                        if (Path.GetFileNameWithoutExtension(file) == "ACC")
                        {
                            Tuple<string, double[]> AccData = ProcessE4File(file);
                            Data.Add(AccData.Item2);
                            Labels.Add(AccData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "BVP")
                        {
                            Tuple<string, double[]> BvpData = ProcessE4File(file);
                            Data.Add(BvpData.Item2);
                            Labels.Add(BvpData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "EDA")
                        {
                            Tuple<string, double[]> EdaData = ProcessE4File(file);
                            Data.Add(EdaData.Item2);
                            Labels.Add(EdaData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "HR")
                        {
                            Tuple<string, double[]> HrData = ProcessE4File(file);
                            Data.Add(HrData.Item2);
                            Labels.Add(HrData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "IBI")
                        {
                            Tuple<string, double[]> IbiData = ProcessE4File(file);
                            Data.Add(IbiData.Item2);
                            Labels.Add(IbiData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "tags")
                        {
                            Tuple<string, double[]> TagData = ProcessE4File(file, true);
                            Data.Add(TagData.Item2);
                            Labels.Add(TagData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "TEMP")
                        {
                            Tuple<string, double[]> TmpData = ProcessE4File(file);
                            Data.Add(TmpData.Item2);
                            Labels.Add(TmpData.Item1);
                        }
                    }
                    Dataset.Add(Tuple.Create(FolderName, Data, Labels));
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
