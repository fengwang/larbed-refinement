#ifndef MCUDA_DIRECT_THICKNESS_DSOIASDFLK49O8UYADFKLJHDLKAJSHSADKLJH498YAFDKJHSF
#define MCUDA_DIRECT_THICKNESS_DSOIASDFLK49O8UYADFKLJHDLKAJSHSADKLJH498YAFDKJHSF

#include <f/pattern/pattern.hpp>

#include <functional>
#include <cassert>

struct cuda_pattern;

cuda_pattern* make_cuda_pattern( unsigned long n, unsigned long ug_size );
void release_cuda_pattern( cuda_pattern* cp );
void cuda_pattern_register_entry( cuda_pattern* cp, unsigned long index, unsigned long dim, unsigned long* ar, double* diag, double* intensity );
void cuda_pattern_update_ug_thickness( cuda_pattern* cp, double* p );
double cuda_pattern_make_residual( cuda_pattern* cp );
double cuda_pattern_make_residual( cuda_pattern* cp );

namespace f
{
    
    struct cuda_direct_thickness
    {
        typedef double                                  value_type;
        typedef std::complex<value_type>                complex_type;
        typedef value_type*                             pointer;
        
        cuda_pattern*                                   cuda_pattern_cache;

        ~cuda_direct_thickness()
        { 
            release_cuda_pattern( cuda_pattern_cache ); 
        }

        cuda_direct_thickness( pattern<value_type> const& pt_ )
        {
            cuda_pattern_cache = make_cuda_pattern( pt.tilt_size, pt.ug_size );

            for ( size_type index = 0; index != pt.tilt_size; ++index )
                cuda_pattern_register_entry( cuda_pattern_cache, index, pt.ar.row(), (pt.ar)[index].data(), (pt.diag)[index].data(), (pt.intensity)[index].data() );
        }

        std::function<value_type(pointer)> make_merit_function()
        {
            return [this]( pointer p )
            {
                cuda_pattern_update_ug_thickness( (*this).cuda_pattern_cache, p );
                return cuda_pattern_make_residual( (*this).cuda_pattern_cache );
            };
        }

        std::function<value_type(pointer)> make_abs_function()
        {
            return [this]( pointer p )
            {
                cuda_pattern_update_ug_thickness( (*this).cuda_pattern_cache, p );
                return cuda_pattern_make_abs_residual( (*this).cuda_pattern_cache );
            };
        }

    };//struct cuda_direct_thickness

}//namespace f

#endif//_CUDA_DIRECT_THICKNESS_DSOIASDFLK49O8UYADFKLJHDLKAJSHSADKLJH498YAFDKJHSF

