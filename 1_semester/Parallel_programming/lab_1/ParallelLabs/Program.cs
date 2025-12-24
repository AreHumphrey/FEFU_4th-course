using System;
using ParallelLabs.Labs.Lab01;
using ParallelLabs.Labs.Lab02;
using ParallelLabs.Labs.Lab03;
using ParallelLabs.Labs.Lab04;
using ParallelLabs.Labs.Lab05;
using ParallelLabs.Labs.Lab06;
using ParallelLabs.Labs.Lab07;
using ParallelLabs.Labs.Lab08;
using ParallelLabs.Labs.Lab09;

namespace ParallelLabs
{
    class Program
    {
        static void Main(string[] args)
        {
            string labName = args.Length > 0 ? args[0] : "Lab01";
            switch (labName)
            {
                case "Lab01": LabProgram.Run(); break;
                case "Lab02": Lab02Program.Run(); break;
                case "Lab03": Lab03Program.Run(); break;
                case "Lab04": Lab04Program.Run(); break;
                case "Lab05": Lab05Program.Run(); break;
                case "Lab06": Lab06Program.Run(); break;
                case "Lab07": Lab07Program.Run(); break;
                case "Lab08": Lab08Program.Run(); break;
                case "Lab09": Lab09Program.Run(); break;
                default:
                    Console.WriteLine($"❌ Неизвестная лабораторная: {labName}");
                    Console.WriteLine("Доступные: Lab01–Lab09");
                    break;
            }
        }
    }
}
