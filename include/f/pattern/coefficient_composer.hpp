#ifndef SRLKAITHWSTFQACLNKPTUULCJSEYKYCLRLAHPBDYPAFOJEHHEFRNSUWHYYKSRSDWCTYCPXEFG
#define SRLKAITHWSTFQACLNKPTUULCJSEYKYCLRLAHPBDYPAFOJEHHEFRNSUWHYYKSRSDWCTYCPXEFG

#include <f/matrix/matrix.hpp>
#include <f/polynomial/term.hpp>
#include <f/polynomial/polynomial.hpp>
#include <f/pattern/elementary_pattern.hpp>

#include <vector>

namespace f
{
    struct coefficient_composer
    {
        typedef elementary_pattern::size_type               size_type;
        typedef elementary_pattern::simple_term_type        term_type;
        typedef term_type::multi_symbol_type                multi_symbol_type;
        typedef std::vector<multi_symbol_type>              multi_symbol_array_type;
        typedef std::vector<term_type>                      term_array_type;
        typedef elementary_pattern::value_type              value_type;
        typedef std::vector<value_type>                     value_array_type;
        typedef matrix<value_type>                          value_matrix_type;
        typedef elementary_pattern::simple_polynomial_type  simple_polynomial_type;
        typedef std::vector<simple_polynomial_type>         simple_polynomial_array_type;

        simple_polynomial_type                              polynomial_terms_accumulator;
        simple_polynomial_array_type                        real_polynomial_array;
        simple_polynomial_array_type                        imag_polynomial_array;
        value_array_type                                    intensity_array;

        term_array_type                                     unique_term_array;
        size_type                                           unique_term_number;

        value_matrix_type                                   vx;
        value_matrix_type                                   vy;

        void reset()
        {
            polynomial_terms_accumulator.clear();
            real_polynomial_array.clear();
            imag_polynomial_array.clear();
            intensity_array.clear();
            unique_term_array.clear();
            unique_term_number = 0;
        }

        // conceptually, y = real_p * real_p + imag_p * imag_p
        void register_real_imag_intensity( simple_polynomial_type const& real_p_, simple_polynomial_type const& imag_p_, value_type y_ )
        {
            polynomial_terms_accumulator += abs( real_p_ );
            polynomial_terms_accumulator += abs( imag_p_ );

            real_polynomial_array.push_back( real_p_ );
            imag_polynomial_array.push_back( imag_p_ );

            intensity_array.push_back( y_ );
        }

        void process_unique_terms()
        {
            //find out how many terms involved
            unique_term_number = polynomial_terms_accumulator.size();
            //put all terms into a vector
            for ( auto const& element : polynomial_terms_accumulator.collection )
                unique_term_array.push_back( element );
                //unique_term_array.push_back( element.symbols );
        }

        void process_vx()
        {
            size_type const row = intensity_array.size();
            size_type const col = unique_term_number + unique_term_number;

            vx.resize( row, col );

            for ( size_type r = 0; r != row; ++r )
            {
                //fill real coefficient
                auto const& real_p = real_polynomial_array[r];
                for ( size_type c = 0; c != unique_term_number; ++c )
                {
                    auto const& the_term = unique_term_array[c];
                    vx[r][c] = value_type{0};
                    auto itor = real_p.collection.find( the_term );
                    if ( itor != real_p.collection.end() )
                        vx[r][c] = (*itor).coefficient;
                }

                //fill imaginary coefficient
                auto const& imag_p = imag_polynomial_array[r];
                for ( size_type c = 0; c != unique_term_number; ++c )
                {
                    auto const& the_term = unique_term_array[c];
                    vx[r][unique_term_number+c] = value_type{0};
                    auto itor = imag_p.collection.find( the_term );
                    if ( itor != imag_p.collection.end() )
                        vx[r][c+unique_term_number] = (*itor).coefficient;
                }
            }
        }

        void process_vy()
        {
            vy.resize( intensity_array.size(), 1 );
            std::copy( intensity_array.begin(), intensity_array.end(), vy.begin() );
        }


        void process()
        {
            process_unique_terms();
            process_vx();
            process_vy();
        }

        friend std::ostream& operator << ( std::ostream& os, coefficient_composer const& cc )
        {
            //output terms
            os << "the unique terms:\n" << cc.polynomial_terms_accumulator << "\n";
            //output real_p, imag_p
            size_type const& r = cc.real_polynomial_array.size();
            os << "the real -- imag polynomials:\n";
            for ( size_type r_ = 0; r_ != r; ++r_ )
                os << cc.real_polynomial_array[r_] << "--" << cc.imag_polynomial_array[r_] << "\n";

            os << "the vx is:\n" << cc.vx << "\n";
            os << "the vy is:\n" << cc.vy << "\n";
            //output vx, vy
            return os;
        }
    };//struct coefficient_composer

}//namespace f

#endif//SRLKAITHWSTFQACLNKPTUULCJSEYKYCLRLAHPBDYPAFOJEHHEFRNSUWHYYKSRSDWCTYCPXEFG

