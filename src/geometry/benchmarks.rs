use super::{circle_2d, grid_arange, tank_2d};

/// Create stack of cylinders as in the benchmark of Zhang
pub fn create_cylinders_zhang(spacing: f32) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    // We follow three steps while creating the geometry
    // - Create bottom layer
    // - Create bottom second layer
    // - Create bottom third layer

    // ---------------------------------------
    // Create bottom layer
    // create a cylinder in 2d (that would be a circle),
    let diameter = 0.01; // in meters
    let (xc1, yc1) = circle_2d(
        (diameter / 2. + spacing, diameter / 2. + spacing),
        diameter / 2.,
        spacing,
    );

    // get the number of particles in cylinder to compute the body id later
    let no_of_particles = xc1.len();

    let xc2: Vec<_> = xc1.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc2 = yc1.clone();
    let xc3: Vec<_> = xc2.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc3 = yc1.clone();
    let xc4: Vec<_> = xc3.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc4 = yc1.clone();
    let xc5: Vec<_> = xc4.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc5 = yc1.clone();
    let xc6: Vec<_> = xc5.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc6 = yc1.clone();
    let x_layer_1 = [&xc1[..], &xc2[..], &xc3[..], &xc4[..], &xc5[..], &xc6[..]].concat();
    let y_layer_1 = [&yc1[..], &yc2[..], &yc3[..], &yc4[..], &yc5[..], &yc6[..]].concat();
    // ---------------------------------------

    // ---------------------------------------
    // Create second bottom layer
    let (xc1, yc1) = circle_2d(
        (diameter + spacing, 1.5 * diameter + spacing),
        diameter / 2.,
        spacing,
    );
    let xc2: Vec<_> = xc1.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc2 = yc1.clone();
    let xc3: Vec<_> = xc2.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc3 = yc1.clone();
    let xc4: Vec<_> = xc3.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc4 = yc1.clone();
    let xc5: Vec<_> = xc4.iter().map(|&i| i + diameter + spacing / 2.).collect();
    let yc5 = yc1.clone();
    let x_layer_2 = [&xc1[..], &xc2[..], &xc3[..], &xc4[..], &xc5[..]].concat();
    let y_layer_2 = [&yc1[..], &yc2[..], &yc3[..], &yc4[..], &yc5[..]].concat();

    // ---------------------------------------
    // Create third bottom layer
    let x_layer_3 = x_layer_1.clone();
    let y_layer_3 = y_layer_1
        .iter()
        .map(|&i| i + 2. * diameter + spacing / 2.)
        .collect::<Vec<_>>();

    // ---------------------------------------
    // ---------------------------------------
    // Create fourth bottom layer
    let x_layer_4 = x_layer_2.clone();
    let y_layer_4 = y_layer_2
        .iter()
        .map(|&i| i + 2. * diameter + spacing / 2.)
        .collect::<Vec<_>>();

    // Create fifth bottom layer
    let x_layer_5 = x_layer_3.clone();
    let y_layer_5 = y_layer_3
        .iter()
        .map(|&i| i + 2. * diameter + spacing / 2.)
        .collect::<Vec<_>>();

    // Create sixth bottom layer
    let x_layer_6 = x_layer_4.clone();
    let y_layer_6 = y_layer_4
        .iter()
        .map(|&i| i + 2. * diameter + spacing / 2.)
        .collect::<Vec<_>>();

    let x_all = [
        &x_layer_1[..],
        &x_layer_2[..],
        &x_layer_3[..],
        &x_layer_4[..],
        &x_layer_5[..],
        &x_layer_6[..],
    ]
        .concat();
    let y_all = [
        &y_layer_1[..],
        &y_layer_2[..],
        &y_layer_3[..],
        &y_layer_4[..],
        &y_layer_5[..],
        &y_layer_6[..],
    ]
        .concat();

    // create body id of each cylinder
    let no_of_cylinders = 3 * 6 + 3 * 5;
    let mut b_id = vec![];

    for i in 0..no_of_cylinders {
        b_id.extend_from_slice(&vec![i; no_of_particles]);
    }

    (x_all, y_all, b_id)
}

/// Create geometry as in the benchmark of Zhang solid bodies
pub fn create_zhang_geometry(spacing: f32) -> (Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) {
    // get the x, y and body id vectors
    let (xc, yc, bid) = create_cylinders_zhang(spacing);
    // create the tank
    let layers = 3;
    // create a tank with 26 cm length, 26 cm height
    let (xt, yt) = tank_2d(
        0.0,
        0.26 + spacing / 2.,
        spacing,
        0.0,
        0.26 + spacing / 2.,
        spacing,
        layers,
        true,
    );

    (xc, yc, bid, xt, yt)
}

/// 2d breaking dam geometry
pub fn create_2d_breaking_dam_geometry(
    spacing: Option<f32>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // switch back to default if nothing is specified
    let spacing = spacing.unwrap_or(0.001);

    // water column is 0.05717 m high. Ceate a grid of fluid block
    let add = 0.007;
    let (xf, yf) = grid_arange(
        0.0 + add,
        0.05717 + spacing / 2. + add,
        spacing,
        0.0 + add,
        0.05717 + spacing / 2. + add,
        spacing,
    );
    // create the tank
    let (xt, yt) = tank_2d(0.0, 0.2, spacing, 0.0, 0.2, spacing, 2, true);

    (xf, yf, xt, yt)
}
