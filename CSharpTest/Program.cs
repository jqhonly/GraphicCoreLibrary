using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GCLSharp;

namespace CSharpTest
{
    class Program
    {
        static void Main(string[] args)
        {
            byte [] yv12=new byte[1280*720*3/2];
            byte [] rgba32=new byte[1280*720*4];
            
            GCLSharp.ColorTransSharp ctrans = new ColorTransSharp(1280, 720);
            Stopwatch sw=new Stopwatch();
            for (int j = 0; j < 20; j++)
            {
                for (int i = 0; i < 1280 * 720 * 3 / 2; i++)
                {
                    yv12[i] = Convert.ToByte(i*j % 255);
                }
                sw.Reset();
                sw.Start();
                Console.WriteLine("CUDA Error: " + ctrans.Managed_ColorTrans_YV12toARGB32(yv12, rgba32, 0));
                sw.Stop();
                Console.WriteLine(sw.ElapsedMilliseconds+ "ms");
            }
           
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(rgba32[i]);
            }
            Console.ReadLine();
            int a = 1;
        }
    }
}
