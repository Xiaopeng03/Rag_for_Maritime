% 读取JSON文件
filename = 'your_file.json';  % 替换为你的JSON文件名
fid = fopen(filename, 'r');
raw = fread(fid, inf);
str = char(raw');
fclose(fid);

% 解析JSON
data = jsondecode(str);

% 修改参数
data.a.a1 = 100;
data.a.a2 = 200;
data.a.a3 = 300;

% 转换回JSON格式
jsonStr = jsonencode(data);

% 格式化JSON（可选，使其更易读）
jsonStr = strrep(jsonStr, ',', sprintf(',\n'));
jsonStr = strrep(jsonStr, '{', sprintf('{\n'));
jsonStr = strrep(jsonStr, '}', sprintf('\n}'));

% 保存JSON文件
fid = fopen(filename, 'w');
fwrite(fid, jsonStr, 'char');
fclose(fid);

disp('JSON文件已成功修改并保存');
如果你想要更简洁的版本（不格式化）：

% 读取JSON
filename = 'your_file.json';
fid = fopen(filename, 'r');
raw = fread(fid, inf);
str = char(raw');
fclose(fid);

% 修改数据
data = jsondecode(str);
data.a.a1 = 100;
data.a.a2 = 200;
data.a.a3 = 300;

% 保存JSON
fid = fopen(filename, 'w');
fwrite(fid, jsonencode(data), 'char');
fclose(fid);