
@testset "Dynamic" begin
    @testset "Check Dimensions" begin
        real_k = 3
        real_n = 3
        real_p = [1, 2, 3]
        real_nreps = 4
        real_T = 10

        Y = ones(Float64, real_T, real_n * real_nreps)
        F = [ones(Float64, real_n, real_p[j]) for j = 1:real_k]
        G = [ones(Float64, real_p[j], real_p[j]) for j = 1:real_k]

        n, nreps, p, T, k, index_map = DynamicMixtures.check_dimensions(Y, F, G)

        @test k == real_k
        @test n == real_n
        @test p == real_p
        @test nreps == real_nreps
        @test T == real_T

        @test index_map[1] == 1:3
        @test index_map[2] == 4:6
        @test index_map[3] == 7:9
        @test index_map[4] == 10:12

        #TODO: Add tests checking the throws
    end
end
