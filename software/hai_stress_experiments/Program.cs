using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MMIVR.BiosensorFramework.MachineLearningUtilities;

namespace hai_stress_experiments
{
    class Program
    {
        static void Main(string[] args)
        {
            string TopDir = @"C:\GitHub\hai_stress\data\raw_data";
            var Dataset = DataImport.LoadE4Dataset(TopDir);
        }
    }
}
