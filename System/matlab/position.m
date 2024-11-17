function [pixel_coords, birth_range, persistence_range] = position(flatten_index, minBD, maxBD, image_size)
    % 计算范围长度
    range_length = maxBD - minBD;

    % 计算每个像素点的分辨率
    pixel_resolution = range_length / image_size;

    % 将flatten后的向量索引转换为PI的像素点
    pixel_x = mod(flatten_index - 1, image_size);
    pixel_y = floor((flatten_index - 1) / image_size);

    % 计算横坐标（birth）范围
    birth_min = minBD + pixel_x * pixel_resolution;
    birth_max = minBD + (pixel_x + 1) * pixel_resolution;

    % 计算纵坐标（persistence）范围，注意纵坐标方向相反
    persistence_min = minBD + (image_size - 1 - pixel_y) * pixel_resolution;
    persistence_max = minBD + (image_size - pixel_y) * pixel_resolution;

    pixel_coords = [pixel_x, pixel_y];
    birth_range = [birth_min, birth_max];
    persistence_range = [persistence_min, persistence_max];
end