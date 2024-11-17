% 示例使用
flatten_index = 1389;
minBD = -0.1;
maxBD = 1.5;
image_size = 40;

[pixel_coords, birth_range, persistence_range] = position(flatten_index, minBD, maxBD, image_size);

disp(['Flatten index ', num2str(flatten_index), ' corresponds to pixel: (', num2str(pixel_coords(1)), ', ', num2str(pixel_coords(2)), ')']);
disp(['Birth range: [', num2str(birth_range(1)), ', ', num2str(birth_range(2)), ']']);
disp(['Persistence range: [', num2str(persistence_range(1)), ', ', num2str(persistence_range(2)), ']']);
