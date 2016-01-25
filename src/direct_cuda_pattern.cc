#include <f/optimization/nonlinear/steepest_descent.hpp>
#include <f/pattern/pattern.hpp>
#include <f/cuda_pattern/cuda_pattern.hpp>
#include <f/date/date_to_string.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include <f/pattern_refinement/reflection_residual.hpp>

#include <vector>
#include <iostream>
#include <complex>
#include <string>

int main()
{
    auto pt = f::make_pattern( "testdata/new_txt", std::complex<double>{0.0, 12.0} );
    f::cuda_pattern cpt{ pt, 3 };

    std::vector<double> ug_initial;
    ug_initial.resize( pt.ug_size*2 + 1 );
    std::fill( ug_initial.begin(), ug_initial.end(), 0.00 );

    unsigned long const diffraction_dim = 121;


	f::variate_generator<double> vg( 0.0, 0.01 );
	std::generate( ug_initial.begin(), ug_initial.end(), vg );

    f::reflection_residual rr{ std::string{"./matrix/beam/SrTiO3.txt"}, std::string{"./matrix/intensity/SrTiO3.txt"}, pt.ug_size };
    auto const& reflection_merit = rr.make_normal_residual();

    auto const& merit_function_ = cpt.make_abs_function(); 

    unsigned long const unknowns = pt.ug_size*2+1;
    unsigned long const zos = pt.ug_size - 1;
    auto const& merit_function = [&reflection_merit, &merit_function_, unknowns, zos]( double* x )
    {
        double const res = std::inner_product( x, x+unknowns, x, 0.0 );

        double rms = 0.0;
        double* itor = x + 3;
        for ( unsigned long idx = 0; idx != zos; ++idx )
        {
            double rx = *itor;
            rms += rx*rx;
            itor += 2;
        }

        return reflection_merit( x ) + merit_function_( x ) + res * 100.9 + rms * 1009.00;
    };

    f::simple_steepest_descent<double> sd( merit_function, pt.ug_size * 2 + 1 );
    sd.config_initial_guess( ug_initial.begin() );
    sd.config_total_steps( 2000 );
    sd.config_eps( 1.0e-10 );

    std::string file_name = f::date_to_string() + std::string{"_direct_cuda_pattern.dat"};
    std::string abs_file_name = f::date_to_string() + std::string{"_direct_cuda_pattern_R.dat"};

    auto const& abs_function = cpt.make_abs_function();

    auto on_iteration_over = [ &file_name, &abs_file_name, unknowns, &merit_function_ ]( double residual, double* current_solution )
    {
        std::ofstream ofs1( file_name.c_str(), std::fstream::app );
        ofs1 << residual << "\t";
        std::copy( current_solution, current_solution+unknowns, std::ostream_iterator<double>( ofs1, "\t" ) );
        ofs1 << "\n";
        ofs1.close();

        std::ofstream ofs2( abs_file_name.c_str(), std::fstream::app );
        double const abs_residual =  merit_function_( current_solution );
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

    return 0;
}

