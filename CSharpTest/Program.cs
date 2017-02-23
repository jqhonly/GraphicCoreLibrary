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
            //byte[] yv12 = new byte[1280*720*3/2];
            //byte[] rgba32 = new byte[1280*720*4];

            //ColorTransSharp ctrans = new ColorTransSharp(1280, 720, 1);
            //Stopwatch sw=new Stopwatch();
            //for (int j = 0; j < 10; j++)
            //{
            //    for (int i = 0; i < 1280 * 720 * 3 / 2; i++)
            //    {
            //        yv12[i] = Convert.ToByte(i*j % 255);
            //    }
            //    sw.Reset();
            //    sw.Start();
            //    Console.WriteLine("CUDA Error: " + ctrans.Managed_ColorTrans_YV12toARGB32_RetineX(yv12, rgba32));
            //    sw.Stop();
            //    Console.WriteLine(sw.ElapsedMilliseconds+ "ms");
            //}

            //for (int i = 0; i < 10; i++)
            //{
            //    Console.WriteLine(rgba32[i]);
            //}
            
            CameraManageSharp cm=new CameraManageSharp("192.168.0.68", 8000, "admin", "hk123456");
            bool x=cm.login_managed();
            x = cm.play_managed(IntPtr.Zero);
            for (int i = 0; i < 100; i++)
            {
                byte[] data = cm.get_frame_managed();
            }
            cm.stop_managed();
            cm.logout_managed();
            Console.ReadLine();
            int a = 1;
        }
    }
}
