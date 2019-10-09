#include <f/optimization/nonlinear/steepest_descent.hpp>
#include <f/pattern/pattern.hpp>
#include <f/cuda_pattern/cuda_pattern.hpp>
#include <f/date/date_to_string.hpp>

#include <vector>
#include <iostream>
#include <complex>
#include <string>

int main()
{
    int const thickness_ = 14.5;
    //int gpu_id = 1;
    int gpu_id = 0;
    double const init_thickness = thickness_;
    std::complex<double> const thickness{ 0.0, init_thickness  };
    auto pt = f::make_simulated_pattern("testdata/new_txt", thickness);

#if 0
    auto pt = f::make_pattern("testdata/new_txt", thickness);
    {   //update fake ug
        f:matrix<double> ug;
        ug.load( "./testdata/new_txt/SrTiO3_full.txt" );
        pt.update_ug( ug );
    }
    pt.simulate_intensity();
#endif

    f::cuda_pattern cpt{ pt, gpu_id };

    std::vector<double> ug_initial;
    ug_initial.resize( pt.ug_size*2 + 1 );
    std::fill( ug_initial.begin(), ug_initial.end(), 0.01 );
    ug_initial[0] = 1.0;
    ug_initial[1] = 0.0;
    ug_initial[pt.ug_size*2] = init_thickness;

    auto const& merit_function = cpt.make_merit_function();

    f::simple_steepest_descent<double> sd( merit_function, pt.ug_size * 2 + 1 );
    sd.config_initial_guess( ug_initial.begin() );
    sd.config_total_steps( 1000 );
    sd.config_eps( 1.0e-10 );

    std::string tk = std::to_string(thickness_);
    std::string file_name = f::date_to_string() + std::string{"-"} + tk + std::string{"_direct_cuda_pattern.dat"};
    std::string abs_file_name = f::date_to_string() + std::string{"-"} + tk +  std::string{"_direct_cuda_pattern_abs.dat"};

    const unsigned long unknowns = pt.ug_size * 2 + 1;

    auto const& abs_function = cpt.make_abs_function();

    auto on_iteration_over = [ &file_name, &abs_file_name, unknowns, &abs_function ]( double residual, double* current_solution )
    {
        std::ofstream ofs1( file_name.c_str(), std::fstream::app );
        ofs1 << residual << "\t";
        std::copy( current_solution, current_solution+unknowns, std::ostream_iterator<double>( ofs1, "\t" ) );
        ofs1 << "\n";
        ofs1.close();

        std::ofstream ofs2( abs_file_name.c_str(), std::fstream::app );
        double const abs_residual =  abs_function( current_solution );
        ofs2 << abs_residual << "\t";
        std::copy( current_solution, current_solution+unknowns, std::ostream_iterator<double>( ofs2, "\t" ) );
        ofs2 << "\n";
        ofs2.close();
    };

    on_iteration_over( merit_function( ug_initial.data() ), ug_initial.data() );

    sd.config_iteration_function( on_iteration_over );

    sd( ug_initial.begin() );

    f::matrix<std::complex<double> > rug{ ug_initial.size() / 2, 1 };
    std::copy( ug_initial.begin(), ug_initial.end(), reinterpret_cast<double*>(rug.data()) );
    std::cout << "\nsolution:\n" << rug << "\n";

    cpt.dump_a();

    return 0;
}

