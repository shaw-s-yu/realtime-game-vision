using System;
using System.Collections.Generic;
using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace RealtimeGameVisionConfig
{
    public class ConfigEditor
    {
        public string ConfigPath { get; }
        private Dictionary<string, object> root;
        private readonly ISerializer serializer;
        private readonly IDeserializer deserializer;

        public ConfigEditor(string path)
        {
            ConfigPath = Path.GetFullPath(path);
            serializer = new SerializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .ConfigureDefaultValuesHandling(DefaultValuesHandling.OmitNull)
                .Build();
            deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .IgnoreUnmatchedProperties()
                .Build();
            Load();
        }

        public Dictionary<string, object> Load()
        {
            if (!File.Exists(ConfigPath)) { root = new Dictionary<string, object>(); return root; }
            var yaml = File.ReadAllText(ConfigPath);
            try {
                root = deserializer.Deserialize<Dictionary<string, object>>(yaml) ?? new Dictionary<string, object>();
            } catch { root = new Dictionary<string, object>(); }
            return root;
        }

        public object Get(string dotPath, object defaultVal = null)
        {
            var parts = dotPath.Split('.');
            object cur = root;
            foreach (var p in parts)
            {
                if (cur is Dictionary<object, object> d1 && d1.TryGetValue(p, out var v1)) cur = v1;
                else if (cur is Dictionary<string, object> d2 && d2.TryGetValue(p, out var v2)) cur = v2;
                else return defaultVal;
            }
            return cur ?? defaultVal;
        }

        public void Set(string dotPath, object value)
        {
            var parts = dotPath.Split('.');
            var cur = root as IDictionary<string, object> ?? new Dictionary<string, object>();
            // ensure root type is string-keyed
            root = ConvertToStringDict(root);
            IDictionary<string, object> node = root;
            for (int i = 0; i < parts.Length - 1; i++)
            {
                var p = parts[i];
                if (!node.ContainsKey(p) || !(node[p] is IDictionary<string, object> || node[p] is Dictionary<object, object>))
                {
                    node[p] = new Dictionary<string, object>();
                }
                node = ConvertToStringDict(node[p]);
            }
            node[parts[^1]] = value;
        }

        private Dictionary<string, object> ConvertToStringDict(object obj)
        {
            if (obj is Dictionary<string, object> ds) return ds;
            var nd = new Dictionary<string, object>();
            if (obj is Dictionary<object, object> doo)
            {
                foreach (var kv in doo) nd[kv.Key.ToString()] = kv.Value is Dictionary<object, object> ? ConvertToStringDict(kv.Value) : kv.Value;
            }
            return nd;
        }

        public void Save()
        {
            var yaml = serializer.Serialize(root);
            File.WriteAllText(ConfigPath, yaml);
        }
    }
}
